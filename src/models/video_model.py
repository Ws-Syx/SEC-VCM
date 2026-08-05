import torch, math
from torch import nn
import torch.nn.functional as F

from .common_model import CompressionModel, LowerBound
from ..layers.layers import (subpel_conv1x1, conv3x3, subpel_conv3x3,
                              ResidualBlock, ResidualBlockWithStride, ResidualBlockUpsample)
from ..utils.stream_helper import get_state_dict


# =========================================================================
#  Low-level network components
# =========================================================================

backward_grid = [{} for _ in range(9)]


def torch_warp(feature, flow):
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    if str(flow.size()) not in backward_grid[device_id]:
        N, _, H, W = flow.size()
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=feature.device, dtype=feature.dtype).view(
            1, 1, 1, W).expand(N, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=feature.device, dtype=feature.dtype).view(
            1, 1, H, 1).expand(N, -1, -1, W)
        backward_grid[device_id][str(flow.size())] = torch.cat([tensor_hor, tensor_ver], 1)

    flow = torch.cat([flow[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)

    grid = (backward_grid[device_id][str(flow.size())] + flow)
    return torch.nn.functional.grid_sample(input=feature,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)


def flow_warp(im, flow):
    return torch_warp(im, flow)


def bilinearupsacling(inputfeature):
    ih, iw = inputfeature.size()[2], inputfeature.size()[3]
    return F.interpolate(inputfeature, (ih * 2, iw * 2), mode='bilinear', align_corners=False)


def bilineardownsacling(inputfeature):
    ih, iw = inputfeature.size()[2], inputfeature.size()[3]
    return F.interpolate(inputfeature, (ih // 2, iw // 2), mode='bilinear', align_corners=False)


class ResBlock(nn.Module):
    def __init__(self, channel, slope=0.01, start_from_relu=True, end_with_relu=False,
                 bottleneck=False):
        super().__init__()
        self.relu = nn.LeakyReLU(negative_slope=slope)
        if slope < 0.0001:
            self.relu = nn.ReLU()
        if bottleneck:
            self.conv1 = nn.Conv2d(channel, channel // 2, 3, padding=1)
            self.conv2 = nn.Conv2d(channel // 2, channel, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
            self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.first_layer = self.relu if start_from_relu else nn.Identity()
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return x + out


class MEBasic(nn.Module):
    def __init__(self, layername=None):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([MEBasic() for _ in range(4)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1_list = [im1_pre]
        im2_list = [im2_pre]
        for level in range(self.L - 1):
            im1_list.append(F.avg_pool2d(im1_list[level], kernel_size=2, stride=2))
            im2_list.append(F.avg_pool2d(im2_list[level], kernel_size=2, stride=2))

        shape_fine = im2_list[self.L - 1].size()
        zero_shape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flow = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        for level in range(self.L):
            flow_up = bilinearupsacling(flow) * 2.0
            img_index = self.L - 1 - level
            flow = flow_up + \
                self.moduleBasic[level](torch.cat([im1_list[img_index],
                                                   flow_warp(im2_list[img_index], flow_up),
                                                   flow_up], 1))
        return flow


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = torch.mean(x, dim=(-1, -2))
        y = self.fc(y)
        return x * y[:, :, None, None]


class ConvBlockResidual(nn.Module):
    def __init__(self, ch_in, ch_out, se_layer=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            SELayer(ch_out) if se_layer else nn.Identity(),
        )
        self.up_dim = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x) + self.up_dim(x)


class UNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlockResidual(ch_in=in_ch, ch_out=32)
        self.conv2 = ConvBlockResidual(ch_in=32, ch_out=64)
        self.conv3 = ConvBlockResidual(ch_in=64, ch_out=128)

        self.context_refine = nn.Sequential(
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = ConvBlockResidual(ch_in=128, ch_out=64)
        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = ConvBlockResidual(ch_in=64, ch_out=out_ch)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)
        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)
        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


def get_enc_dec_models(input_channel, output_channel, channel):
    enc = nn.Sequential(
        ResidualBlockWithStride(input_channel, channel, stride=2),
        ResidualBlock(channel, channel),
        ResidualBlockWithStride(channel, channel, stride=2),
        ResidualBlock(channel, channel),
        ResidualBlockWithStride(channel, channel, stride=2),
        ResidualBlock(channel, channel),
        conv3x3(channel, channel, stride=2),
    )
    dec = nn.Sequential(
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        subpel_conv1x1(channel, output_channel, 2),
    )
    return enc, dec


def get_hyper_enc_dec_models(y_channel, z_channel):
    enc = nn.Sequential(
        conv3x3(y_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel, stride=2),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel, stride=2),
    )
    dec = nn.Sequential(
        conv3x3(z_channel, y_channel),
        nn.LeakyReLU(),
        subpel_conv1x1(y_channel, y_channel, 2),
        nn.LeakyReLU(),
        conv3x3(y_channel, y_channel * 3 // 2),
        nn.LeakyReLU(),
        subpel_conv1x1(y_channel * 3 // 2, y_channel * 3 // 2, 2),
        nn.LeakyReLU(),
        conv3x3(y_channel * 3 // 2, y_channel * 2),
    )
    return enc, dec


# =========================================================================
#  DMC model components: context + reconstruction (reconstruction only)
# =========================================================================

class FeatureExtractor(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)
        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)
        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)
        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super().__init__()
        self.conv3_up = subpel_conv3x3(channel_in, channel_out, 2)
        self.res_block3_up = ResBlock(channel_out)
        self.conv3_out = nn.Conv2d(channel_out, channel_out, 3, padding=1)
        self.res_block3_out = ResBlock(channel_out)
        self.conv2_up = subpel_conv3x3(channel_out * 2, channel_out, 2)
        self.res_block2_up = ResBlock(channel_out)
        self.conv2_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block2_out = ResBlock(channel_out)
        self.conv1_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block1_out = ResBlock(channel_out)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out
        return context1, context2, context3


class ContextualEncoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N + 3, channel_N, 3, stride=2, padding=1)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.conv2 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.conv3 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(channel_N, channel_M, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.up1 = subpel_conv3x3(channel_M, channel_N, 2)
        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up4 = subpel_conv3x3(channel_N * 2, 32, 2)

    def forward(self, x, context2, context3):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=64, res_channel=32, channel=64):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1)
        self.unet_1 = UNet(channel)
        self.unet_2 = UNet(channel)
        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class SemanticDecoder(nn.Module):
    """Semantic reconstruction decoder: produces perceptually optimized frame via SPDF."""

    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.first_conv = nn.Conv2d(channel_M, channel_N, kernel_size=3, padding=1, stride=1)
        self.res1 = ResBlock(channel_N, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up1 = subpel_conv3x3(channel_N, channel_N, 2)
        self.res2 = ResBlock(channel_N, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.res3 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.res4 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up4 = subpel_conv3x3(channel_N * 2, 32, 2)

    def forward(self, latent, context2, context3):
        feature = self.first_conv(latent)
        out16 = self.res1(feature)
        out8 = self.res2(self.up1(out16))
        out4 = self.res3(torch.cat([self.up2(out8), context3], dim=1))
        out2 = self.res4(torch.cat([self.up3(out4), context2], dim=1))
        f = self.up4(out2)
        return f, out2, out4, out8, out16


class GatedGeneration(nn.Module):
    """SPDF (Semantic Prior guided Decoder Fusion): gated fusion of semantic + native features."""

    def __init__(self, ctx_channel=64, res_channel=32, channel=64):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1)
        self.unet_1 = UNet(channel)
        self.unet_2 = UNet(channel)

        self.factor_generator = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            ResBlock(channel=channel, bottleneck=True),
            nn.Sigmoid(),
        )

        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res, recon_feature):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)

        alpha = self.factor_generator(torch.cat((feature, recon_feature), dim=1))
        feature = alpha * feature + (1 - alpha) * recon_feature

        recon = self.recon_conv(feature)
        return feature, recon


class DistributionGeneration(nn.Module):
    """Estimate conditional distribution (mean, scale) for entropy modelling.
    Used in training to compute conditional entropy loss between
    latent representations and teacher features."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.resblock = ResBlock(out_channels, bottleneck=True)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.resblock(x)
        means, scales = x.chunk(2, 1)
        return means, scales


# =========================================================================
#  DMC: Deep Motion Compensation video compression model
# =========================================================================

class DMC(CompressionModel):
    def __init__(self, anchor_num=4):
        super().__init__(y_distribution='laplace', z_channel=64, mv_z_channel=64)

        channel_mv = 64
        channel_N = 64
        channel_M = 96

        self.channel_mv = channel_mv
        self.channel_N = channel_N
        self.channel_M = channel_M

        # Motion estimation (SpyNet)
        self.optic_flow = ME_Spynet()

        # Motion encoder/decoder + hyperprior
        self.mv_encoder, self.mv_decoder = get_enc_dec_models(2, 2, channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N)

        self.mv_y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
        )

        self.mv_y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_mv * 4, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 2, 3, padding=1),
        )

        # Context features
        self.feature_adaptor_I = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(channel_N, channel_N, 1)
        self.feature_extractor = FeatureExtractor(channel_N)
        self.context_fusion_net = MultiScaleContextFusion(channel_N, channel_N)

        # Contextual encoder
        self.contextual_encoder = ContextualEncoder(channel_N=channel_N, channel_M=channel_M)

        self.contextual_hyper_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_M, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        self.contextual_hyper_prior_decoder = nn.Sequential(
            conv3x3(channel_N, channel_M),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M, channel_M, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M, channel_M * 3 // 2),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M * 3 // 2, channel_M * 3 // 2, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M * 3 // 2, channel_M * 2),
        )

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_N, channel_M * 3 // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channel_M * 3 // 2, channel_M * 2, 3, stride=2, padding=1),
        )

        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_M * 5, channel_M * 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, stride=1, padding=1),
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 2, 3, padding=1),
        )

        # Reconstruction decoder (native DCVC-HEM, used for DPB / motion prediction)
        self.contextual_decoder = ContextualDecoder(channel_N=channel_N, channel_M=channel_M)
        self.recon_generation_net = ReconGeneration()

        # Semantic reconstruction (SPDF, used for final output frame)
        self.semantic_decoder = SemanticDecoder(channel_N=channel_N, channel_M=channel_M)
        self.semantic_generation_net = GatedGeneration()

        # Quantization parameters
        self.mv_y_q_basic = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_basic = nn.Parameter(torch.ones((1, channel_M, 1, 1)))
        self.y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.anchor_num = int(anchor_num)

        self._initialize_weights()

        # ======== Training-only modules (set to None for test-only usage) ========
        # Semantic teacher models (ResNet18, Swin, DinoV2) for perceptual losses
        self.resnet18_model = None     # assign for training setup
        self.semantic_model = None     # assign for training setup
        self.dinov2_model = None       # assign for training setup

        # Conditional distribution estimation for entropy loss (teacher <-> latent)
        self.distribution_generation16 = DistributionGeneration(384, 128)   # swin res4 (384ch) -> latent out16 (64ch) -> (64, 64) mean+scale
        self.distribution_generation8 = DistributionGeneration(192, 128)    # swin res3 (192ch) -> latent out8  (64ch) -> (64, 64) mean+scale
        self.distribution_generation4 = DistributionGeneration(96, 256)     # swin res2 (96ch)  -> latent out4  (128ch)-> (128,128) mean+scale
        self.distribution_generation16_reverse = DistributionGeneration(64, 768)   # latent out16 (64ch) -> swin res4 (384ch) -> (384,384) mean+scale
        self.distribution_generation8_reverse = DistributionGeneration(64, 384)    # latent out8  (64ch) -> swin res3 (192ch) -> (192,192) mean+scale
        self.distribution_generation4_reverse = DistributionGeneration(128, 192)   # latent out4  (128ch)-> swin res2 (96ch)  -> (96,96) mean+scale

        # LPIPS (AlexNet) for distortion metric
        self.alexnet_model = None      # assign during training setup (lpips.LPIPS(net='alex'))

        # ======== Loss criteria ========
        self.mse = None                # will be set externally, e.g. nn.MSELoss(reduction='none')
        self.ssim = None               # will be set externally, e.g. SSIM(data_range=1.0)

    # --- Multi-scale context extraction ---

    def multi_scale_feature_extractor(self, dpb):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            feature = self.feature_adaptor_P(dpb["ref_feature"])
        return self.feature_extractor(feature)

    def motion_compensation(self, dpb, mv):
        warpframe = flow_warp(dpb["ref_frame"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb)
        context1 = flow_warp(ref_feature1, mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    # --- Q-scale helpers ---

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scales = ckpt["y_q_scale"]
        mv_y_q_scales = ckpt["mv_y_q_scale"]
        return y_q_scales.reshape(-1), mv_y_q_scales.reshape(-1)

    def get_curr_mv_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.mv_y_q_basic, 0.5)
        return q_basic * q_scale

    def get_curr_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.y_q_basic, 0.5)
        return q_basic * q_scale

    # --- Conditional entropy loss (training only) ---

    @staticmethod
    def get_conditional_entropy(feature, mean, sigma):
        """Compute cross-entropy between feature and conditional Laplace distribution.
        Used during training to align latent representations with teacher features."""
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            raise ValueError("Found NaN or Inf in mean")
        if torch.isnan(sigma).any() or torch.isinf(sigma).any():
            raise ValueError("Found NaN or Inf in sigma")
        if torch.isnan(feature).any() or torch.isinf(feature).any():
            raise ValueError("Found NaN or Inf in feature")

        values = feature - mean
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(torch.zeros_like(sigma), sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        mean_entropy = torch.mean(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return mean_entropy

    # --- Core encode/decode (bitrate estimation only) ---

    def encode_decode(self, x, dpb, mv_y_q_scale=None, y_q_scale=None):
        """Encode-decode one P-frame with bitrate estimation."""
        encoded = self.forward_one_frame(x, dpb,
                                         mv_y_q_scale=mv_y_q_scale,
                                         y_q_scale=y_q_scale)
        result = {
            "dpb": encoded['dpb'],
            "bit": encoded['bit'].item(),
            "bit_y": encoded['bit_y'].item(),
            "bit_z": encoded['bit_z'].item(),
            "bit_mv_y": encoded['bit_mv_y'].item(),
            "bit_mv_z": encoded['bit_mv_z'].item(),
            "decoding_time": 0,
            "x_hat": encoded['dpb']['ref_frame_semantic'],  # semantic output = actual frame
        }
        return result

    def forward_one_frame(self, x, dpb, mv_y_q_scale=None, y_q_scale=None, lmd_index=None):
        ref_frame = dpb["ref_frame"]

        if lmd_index is None:
            curr_mv_y_q = self.get_curr_mv_y_q(mv_y_q_scale)
            curr_y_q = self.get_curr_y_q(y_q_scale)
        else:
            curr_mv_y_q = self.get_curr_mv_y_q(self.mv_y_q_scale[lmd_index])
            curr_y_q = self.get_curr_y_q(self.y_q_scale[lmd_index])

        # Motion estimation
        est_mv = self.optic_flow(x, ref_frame)
        mv_y = self.mv_encoder(est_mv)
        mv_y = mv_y / curr_mv_y_q
        mv_z = self.mv_hyper_prior_encoder(mv_y)
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            ref_mv_y = torch.zeros_like(mv_y)
        mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
        mv_q_step, mv_scales, mv_means = self.mv_y_prior_fusion(mv_params).chunk(3, 1)

        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_dual_prior(
            mv_y, mv_means, mv_scales, mv_q_step, self.mv_y_spatial_prior)
        mv_y_hat = mv_y_hat * curr_mv_y_q

        # Motion compensation
        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, warp_frame = self.motion_compensation(dpb, mv_hat)

        # Contextual encoding
        y = self.contextual_encoder(x, context1, context2, context3)
        y = y / curr_y_q
        z = self.contextual_hyper_prior_encoder(y)
        z_hat = self.quant(z)
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)

        ref_y = dpb["ref_y"]
        if ref_y is None:
            ref_y = torch.zeros_like(y)
        params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_y_q

        # Reconstruction (native DCVC-HEM, used for DPB / motion prediction)
        recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)

        # Semantic reconstruction (SPDF, used for final output frame)
        semantic_feature, out2, out4, out8, out16 = self.semantic_decoder(y_hat, context2, context3)
        _, semantic_image = self.semantic_generation_net(semantic_feature, context1, feature)

        # --- Distortion losses ---
        B, _, H, W = x.size()
        pixel_num = H * W

        # MSE / SSIM (if criterion modules are set)
        if self.mse is not None and self.ssim is not None:
            mse = self.mse(x, recon_image)
            mse_semantic = self.mse(x, semantic_image)
            ssim = self.ssim(x, recon_image)
            ssim_semantic = self.ssim(x, semantic_image)
            me_mse = self.mse(x, warp_frame)
            mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num
            mse_semantic = torch.sum(mse_semantic, dim=(1, 2, 3)) / pixel_num
            me_mse = torch.sum(me_mse, dim=(1, 2, 3)) / pixel_num
        else:
            mse = torch.tensor(0., device=x.device)
            mse_semantic = torch.tensor(0., device=x.device)
            ssim = torch.tensor(0., device=x.device)
            ssim_semantic = torch.tensor(0., device=x.device)
            me_mse = torch.tensor(0., device=x.device)

        # LPIPS (AlexNet)
        if self.alexnet_model is not None:
            def renorm_for_lpips(img):
                return torch.clip(img * 2 - 1, min=-1.0, max=1.0)
            lpips_alexnet = self.alexnet_model(renorm_for_lpips(x), renorm_for_lpips(recon_image))
        else:
            lpips_alexnet = torch.tensor(0., device=x.device)

        # --- Perceptual losses (teacher models) ---
        # Swin (Mask2Former backbone)
        if self.semantic_model is not None:
            self.semantic_model.eval()
            perception_features_swin = self.semantic_model.backbone(x)
            perception_features_swin_hat = self.semantic_model.backbone(semantic_image)
            mse_swin_res2 = torch.mean(self.mse(perception_features_swin['res2'], perception_features_swin_hat['res2']))
            mse_swin_res3 = torch.mean(self.mse(perception_features_swin['res3'], perception_features_swin_hat['res3']))
            mse_swin_res4 = torch.mean(self.mse(perception_features_swin['res4'], perception_features_swin_hat['res4']))
            mse_swin_res5 = torch.mean(self.mse(perception_features_swin['res5'], perception_features_swin_hat['res5']))
            lpips_swin = (mse_swin_res2 + mse_swin_res3) / 2.0
        else:
            lpips_swin = torch.tensor(0., device=x.device)

        # DinoV2
        if self.dinov2_model is not None:
            self.dinov2_model.eval()
            perception_features_dinov2 = self.dinov2_model(x)
            perception_features_dinov2_hat = self.dinov2_model(semantic_image)
            mse_dinov2_coarse = torch.mean(self.mse(perception_features_dinov2['coarse'], perception_features_dinov2_hat['coarse']))
            mse_dinov2_fine = torch.mean(self.mse(perception_features_dinov2['fine'], perception_features_dinov2_hat['fine']))
            lpips_dinov2 = (mse_dinov2_coarse + mse_dinov2_fine) / 2.0
        else:
            lpips_dinov2 = torch.tensor(0., device=x.device)

        # ResNet18
        if self.resnet18_model is not None:
            self.resnet18_model.eval()
            perception_features_cnn = self.resnet18_model(x)
            perception_features_cnn_hat = self.resnet18_model(semantic_image)
            mse_cnn_res2 = torch.mean(self.mse(perception_features_cnn['res2'], perception_features_cnn_hat['res2']))
            mse_cnn_res3 = torch.mean(self.mse(perception_features_cnn['res3'], perception_features_cnn_hat['res3']))
            mse_cnn_res4 = torch.mean(self.mse(perception_features_cnn['res4'], perception_features_cnn_hat['res4']))
            mse_cnn_res5 = torch.mean(self.mse(perception_features_cnn['res5'], perception_features_cnn_hat['res5']))
            lpips_cnn = (mse_cnn_res2 + mse_cnn_res3) / 2.0
        else:
            lpips_cnn = torch.tensor(0., device=x.device)

        # --- Conditional entropy loss (training only) ---
        if (self.distribution_generation4 is not None and
                self.distribution_generation8 is not None and
                self.distribution_generation16 is not None):
            # Forward: teacher -> latent
            means4, scales4 = self.distribution_generation4(perception_features_swin["res2"])
            means8, scales8 = self.distribution_generation8(perception_features_swin["res3"])
            means16, scales16 = self.distribution_generation16(perception_features_swin["res4"])
            entropy4 = self.get_conditional_entropy(out4, means4, scales4)
            entropy8 = self.get_conditional_entropy(out8, means8, scales8)
            entropy16 = self.get_conditional_entropy(out16, means16, scales16)

            # Reverse: latent -> teacher
            means4_reverse, scales4_reverse = self.distribution_generation4_reverse(out4)
            means8_reverse, scales8_reverse = self.distribution_generation8_reverse(out8)
            means16_reverse, scales16_reverse = self.distribution_generation16_reverse(out16)
            entropy4_reverse = self.get_conditional_entropy(perception_features_swin["res2"], means4_reverse, scales4_reverse)
            entropy8_reverse = self.get_conditional_entropy(perception_features_swin["res3"], means8_reverse, scales8_reverse)
            entropy16_reverse = self.get_conditional_entropy(perception_features_swin["res4"], means16_reverse, scales16_reverse)

            entropy = (entropy4 + entropy8 + entropy16 + entropy4_reverse + entropy8_reverse + entropy16_reverse) / 6.0
        else:
            entropy = torch.tensor(0., device=x.device)
            entropy4 = torch.tensor(0., device=x.device)
            entropy8 = torch.tensor(0., device=x.device)
            entropy16 = torch.tensor(0., device=x.device)
            entropy4_reverse = torch.tensor(0., device=x.device)
            entropy8_reverse = torch.tensor(0., device=x.device)
            entropy16_reverse = torch.tensor(0., device=x.device)

        # --- Bitrate estimation ---
        if self.training:
            y_for_bit = self.add_noise(y_res)
            mv_y_for_bit = self.add_noise(mv_y_res)
            z_for_bit = self.add_noise(z)
            mv_z_for_bit = self.add_noise(mv_z)
        else:
            y_for_bit = y_q
            mv_y_for_bit = mv_y_q
            z_for_bit = z_hat
            mv_z_for_bit = mv_z_hat

        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num

        return {
            "dpb": {
                "ref_frame": recon_image, # native, for motion estimation/compensation
                "ref_frame_semantic": semantic_image, # SPDF output, for saving
                "ref_feature": feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
            "bit": bit,
            "bit_y": bit_y,
            "bit_z": bit_z,
            "bit_mv_y": bit_mv_y,
            "bit_mv_z": bit_mv_z,
            # Distortion metrics
            "me_mse": me_mse,
            "mse": mse,
            "mse_semantic": mse_semantic,
            "ssim": ssim,
            "ssim_semantic": ssim_semantic,
            "lpips_alexnet": lpips_alexnet,
            "lpips_swin": lpips_swin,
            "lpips_cnn": lpips_cnn,
            "lpips_dinov2": lpips_dinov2,
            # Entropy
            "conditional_entropy": entropy,
            "conditional_entropy_4": entropy4,
            "conditional_entropy_8": entropy8,
            "conditional_entropy_16": entropy16,
            "conditional_entropy_4_reverse": entropy4_reverse,
            "conditional_entropy_8_reverse": entropy8_reverse,
            "conditional_entropy_16_reverse": entropy16_reverse,
            # Latents (for training)
            "y_res": y_res,
            "y_q": y_q,
            "y_hat": y_hat,
            "mv_y_res": mv_y_res,
            "mv_y_q": mv_y_q,
            "mv_y_hat": mv_y_hat,
            "z": z,
            "z_hat": z_hat,
            "mv_z": mv_z,
            "mv_z_hat": mv_z_hat,
            "scales_hat": scales_hat,
            "mv_scales_hat": mv_scales_hat,
        }

    def forward(self, x, dpb, mv_y_q_scale=None, y_q_scale=None, lmd_index=None, frame_idx=None):
        """Entry point for both training and inference."""
        return self.forward_one_frame(x, dpb, mv_y_q_scale=mv_y_q_scale,
                                      y_q_scale=y_q_scale, lmd_index=lmd_index)
