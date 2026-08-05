"""
Simple Demo: Neural Video Compression (SEC-VCM reconstruction-only)
Usage: python test_video.py --input_dir <dir> --output_dir <dir>
       --intra_model <ckpt> --inter_model <ckpt> [--rate_num N] [--gop G]
"""
import os
import sys
import time
import argparse
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pytorch_msssim import ms_ssim
import lpips

from src.models.video_model import DMC
from src.models.image_model import IntraNoAR
from src.utils.common import create_folder, generate_log_json, dump_json, interpolate_log
from src.utils.stream_helper import get_padding_size, get_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="SEC-VCM Demo: PNG -> PNG + JSON")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input PNG sequence (e.g., 00000.png, 00005.png, ...)')
    parser.add_argument('--output_dir', type=str, default='./output-sequence',
                        help='Output root directory')
    parser.add_argument('--intra_model', type=str, required=True,
                        help='Path to I-frame (intra) model checkpoint')
    parser.add_argument('--inter_model', type=str, required=True,
                        help='Path to P-frame (inter) model checkpoint')
    parser.add_argument('--rate_num', type=int, default=4,
                        help='Number of rate points to test')
    parser.add_argument('--gop', type=int, default=12,
                        help='GOP size (I-frame interval)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA if available')
    parser.add_argument('--frame_num', type=int, default=-1,
                        help='Max frames to process (-1 = all)')
    return parser.parse_args()


def read_image_to_tensor(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype('float64').transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
    return img


def tensor_to_image(tensor):
    """Convert (1, 3, H, W) tensor to (H, W, 3) uint8 numpy array."""
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    return img


def save_image(tensor, path):
    Image.fromarray(tensor_to_image(tensor)).save(path)


def psnr_fn(a, b):
    mse = torch.mean((a - b) ** 2)
    return (20 * torch.log10(1 / torch.sqrt(mse))).item()


def get_input_frames(input_dir):
    """Get sorted list of PNG frame paths."""
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    paths = [os.path.join(input_dir, f) for f in all_files]
    return paths


def main():
    args = parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # --- Load models ---
    print("[INFO] Loading I-frame (Intra) model...")
    i_state_dict = get_state_dict(args.intra_model)
    i_frame_net = IntraNoAR()
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    print("[INFO] Loading P-frame (Inter) model...")
    p_state_dict = get_state_dict(args.inter_model)
    video_net = DMC()
    video_net.load_state_dict(p_state_dict, strict=False)
    video_net = video_net.to(device)
    video_net.eval()

    # --- Get input frames ---
    frame_paths = get_input_frames(args.input_dir)
    if args.frame_num > 0:
        frame_paths = frame_paths[:args.frame_num]
    total_frames = len(frame_paths)
    print(f"[INFO] Found {total_frames} input frames")

    if total_frames == 0:
        print("[ERROR] No PNG files found in input directory!")
        sys.exit(1)

    # --- Probe first frame for dimensions ---
    first_frame = read_image_to_tensor(frame_paths[0])
    frame_h, frame_w = first_frame.shape[2], first_frame.shape[3]

    # --- Get q_scales from checkpoints ---
    i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt(args.intra_model)
    p_frame_y_q_scales, p_frame_mv_y_q_scales = DMC.get_q_scales_from_ckpt(args.inter_model)

    rate_num = min(args.rate_num, len(i_frame_q_scales), len(p_frame_y_q_scales))

    # Interpolate if needed
    if len(i_frame_q_scales) > rate_num:
        i_frame_q_scales = interpolate_log(i_frame_q_scales[-1], i_frame_q_scales[0], rate_num)
    if len(p_frame_y_q_scales) > rate_num:
        p_frame_y_q_scales = interpolate_log(p_frame_y_q_scales[-1], p_frame_y_q_scales[0], rate_num)
    if len(p_frame_mv_y_q_scales) > rate_num:
        p_frame_mv_y_q_scales = interpolate_log(p_frame_mv_y_q_scales[-1], p_frame_mv_y_q_scales[0], rate_num)

    print(f"[INFO] Testing {rate_num} rate points")
    print(f"       I-frame q_scales:  {[f'{q:.3f}' for q in i_frame_q_scales]}")
    print(f"       P-frame y_q:       {[f'{q:.3f}' for q in p_frame_y_q_scales]}")
    print(f"       P-frame mv_y_q:    {[f'{q:.3f}' for q in p_frame_mv_y_q_scales]}")
    print(f"[INFO] GOP size: {args.gop}")

    # --- LPIPS model (once) ---
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # --- Run for each rate index ---
    create_folder(args.output_dir, print_if_create=True)

    for rate_idx in range(rate_num):
        rate_dir = os.path.join(args.output_dir, f'rate_{rate_idx:02d}')
        create_folder(rate_dir)

        print(f"\n{'='*60}")
        print(f"[Rate {rate_idx:02d}/{rate_num}] I-frame q={i_frame_q_scales[rate_idx]:.3f}, "
              f"P-frame y_q={p_frame_y_q_scales[rate_idx]:.3f}, "
              f"mv_y_q={p_frame_mv_y_q_scales[rate_idx]:.3f}")
        print(f"{'='*60}")

        i_frame_q = i_frame_q_scales[rate_idx]
        p_frame_y_q = p_frame_y_q_scales[rate_idx]
        p_frame_mv_y_q = p_frame_mv_y_q_scales[rate_idx]

        frame_types = []
        bits_list = []
        psnrs_list = []
        msssims_list = []
        lpips_list = []

        overall_start = time.time()
        p_frame_count = 0
        p_decode_time_total = 0.0

        with torch.no_grad():
            for frame_idx in range(total_frames):
                frame_start = time.time()
                img_path = frame_paths[frame_idx]
                x = read_image_to_tensor(img_path).to(device)
                pic_h, pic_w = x.shape[2], x.shape[3]

                # Pad to multiple of 64
                pad_l, pad_r, pad_t, pad_b = get_padding_size(pic_h, pic_w)
                x_padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)

                is_iframe = (frame_idx % args.gop == 0)

                if is_iframe:
                    result = i_frame_net.encode_decode(x_padded, i_frame_q)
                    recon_frame = result["x_hat"]
                    dpb = {
                        "ref_frame": recon_frame,
                        "ref_feature": None,
                        "ref_y": None,
                        "ref_mv_y": None,
                    }
                    frame_types.append(0)
                    bits_list.append(result["bit"])
                else:
                    result = video_net.encode_decode(x_padded, dpb,
                                                     mv_y_q_scale=p_frame_mv_y_q,
                                                     y_q_scale=p_frame_y_q)
                    # DPB for next frame: uses native ref_frame for motion prediction
                    dpb = result["dpb"]
                    # Save frame: uses semantic output (x_hat = ref_frame_semantic)
                    recon_frame = result["x_hat"]
                    frame_types.append(1)
                    bits_list.append(result['bit'])
                    p_frame_count += 1
                    p_decode_time_total += result.get('decoding_time', 0)

                # Unpad and clamp
                recon_frame = recon_frame.clamp_(0, 1)
                x_hat = F.pad(recon_frame, (-pad_l, -pad_r, -pad_t, -pad_b))

                # Metrics
                psnr = psnr_fn(x_hat, x)
                msssim_val = ms_ssim(x_hat, x, data_range=1).item()
                lpips_val = lpips_fn(x_hat, x).squeeze().item()
                psnrs_list.append(psnr)
                msssims_list.append(msssim_val)
                lpips_list.append(lpips_val)

                frame_time = time.time() - frame_start

                # Save decoded PNG
                save_path = os.path.join(rate_dir, f'{frame_idx:05d}.png')
                save_image(x_hat, save_path)

                ftype = "I" if is_iframe else "P"
                bpp = bits_list[-1] / (pic_h * pic_w)
                print(f"  Frame {frame_idx:03d} [{ftype}] | "
                      f"bpp={bpp:.4f} | "
                      f"PSNR={psnr:.3f} | "
                      f"LPIPS={lpips_val:.5f} | "
                      f"time={frame_time:.2f}s")

            total_time = time.time() - overall_start
            frame_pixel_num = int(frame_h * frame_w)

            # --- Generate output JSON ---
            log_result = generate_log_json(
                total_frames, frame_types, bits_list, psnrs_list, msssims_list, lpips_list,
                frame_pixel_num, total_time)

            # Add extra info
            log_result['rate_idx'] = rate_idx
            log_result['i_frame_q_scale'] = float(i_frame_q)
            log_result['p_frame_y_q_scale'] = float(p_frame_y_q)
            log_result['p_frame_mv_y_q_scale'] = float(p_frame_mv_y_q)
            log_result['gop'] = args.gop
            log_result['resolution'] = f'{frame_w}x{frame_h}'

            json_path = os.path.join(rate_dir, 'result.json')
            with open(json_path, 'w') as f:
                dump_json(log_result, f, float_digits=6, indent=2)

            # --- Summary ---
            print(f"\n[Rate {rate_idx:02d}] Summary:")
            print(f"  Total frames: {total_frames}  (I:{log_result['i_frame_num']}, P:{log_result['p_frame_num']})")
            print(f"  Avg BPP:   {log_result['ave_all_frame_bpp']:.4f}")
            print(f"  Total time: {total_time:.1f}s ({total_time/total_frames:.2f}s/frame)")
            if p_frame_count > 0:
                print(f"  P-frame avg decode: {p_decode_time_total/p_frame_count*1000:.1f} ms")
            print(f"  Output saved to: {rate_dir}")

        print(f"\n{'='*60}")
        print(f"[DONE] All {rate_num} rate points completed.")
        print(f"       Results saved to: {args.output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
