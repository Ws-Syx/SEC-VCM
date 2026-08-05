import torch
from torch import nn

from src.layers.layers import conv3x3

from .common_model import CompressionModel
from .video_model import UNet, get_enc_dec_models, get_hyper_enc_dec_models
from .common_model import LowerBound
from ..utils.stream_helper import get_state_dict


class IntraNoAR(CompressionModel):
    def __init__(self, N=192, anchor_num=4):
        super().__init__(y_distribution='gaussian', z_channel=N)

        self.enc, self.dec = get_enc_dec_models(3, 16, N)
        self.refine = nn.Sequential(
            UNet(16, 16),
            conv3x3(16, 3),
        )
        self.hyper_enc, self.hyper_dec = get_hyper_enc_dec_models(N, N)
        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(N * 2, N * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, stride=1, padding=1),
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(N * 4, N * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 2, 3, padding=1),
        )

        self.q_basic = nn.Parameter(torch.ones((1, N, 1, 1)))
        self.q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.N = int(N)
        self.anchor_num = int(anchor_num)

        self._initialize_weights()

    def get_curr_q(self, q_scale):
        q_basic = LowerBound.apply(self.q_basic, 0.5)
        return q_basic * q_scale

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        q_scales = ckpt["q_scale"]
        return q_scales.reshape(-1)

    def encode_decode(self, x, q_scale):
        """Encode-decode one I-frame with bitrate estimation (no actual bitstream)."""
        return self.forward(x, q_scale)

    def forward(self, x, q_scale=None):
        curr_q = self.get_curr_q(q_scale)

        y = self.enc(x)
        y = y / curr_q
        z = self.hyper_enc(y)
        z_hat = self.quant(z)

        params = self.hyper_dec(z_hat)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)

        y_hat = y_hat * curr_q
        x_hat = self.refine(self.dec(y_hat))

        # Bitrate estimation (no training noise)
        y_for_bit = y_q
        z_for_bit = z_hat
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)

        _, _, H, W = x.size()
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bits = torch.sum(bpp_y + bpp_z) * pixel_num

        return {
            "x_hat": x_hat,
            "bit": bits.item(),
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }
