import json
import os
import numpy as np


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


def interpolate_log(min_val, max_val, num, decending=True):
    assert max_val > min_val
    assert min_val > 0
    if decending:
        values = np.linspace(np.log(max_val), np.log(min_val), num)
    else:
        values = np.linspace(np.log(min_val), np.log(max_val), num)
    values = np.exp(values)
    return values


def generate_log_json(frame_num, frame_types, bits, psnrs, msssims, lpipses,
                      frame_pixel_num, test_time):
    """Generate per-frame statistics JSON."""
    cur_ave_i_frame_bit = 0
    cur_ave_i_frame_psnr = 0
    cur_ave_i_frame_msssim = 0
    cur_ave_i_frame_lpips = 0
    cur_ave_p_frame_bit = 0
    cur_ave_p_frame_psnr = 0
    cur_ave_p_frame_msssim = 0
    cur_ave_p_frame_lpips = 0
    cur_i_frame_num = 0
    cur_p_frame_num = 0

    for idx in range(frame_num):
        if frame_types[idx] == 0:
            cur_ave_i_frame_bit += bits[idx]
            cur_ave_i_frame_psnr += psnrs[idx]
            cur_ave_i_frame_msssim += msssims[idx]
            cur_ave_i_frame_lpips += lpipses[idx]
            cur_i_frame_num += 1
        else:
            cur_ave_p_frame_bit += bits[idx]
            cur_ave_p_frame_psnr += psnrs[idx]
            cur_ave_p_frame_msssim += msssims[idx]
            cur_ave_p_frame_lpips += lpipses[idx]
            cur_p_frame_num += 1

    log_result = {
        'frame_pixel_num': frame_pixel_num,
        'i_frame_num': cur_i_frame_num,
        'p_frame_num': cur_p_frame_num,
        'ave_i_frame_bpp': cur_ave_i_frame_bit / max(cur_i_frame_num, 1) / frame_pixel_num,
        'ave_i_frame_psnr': cur_ave_i_frame_psnr / max(cur_i_frame_num, 1),
        'ave_i_frame_msssim': cur_ave_i_frame_msssim / max(cur_i_frame_num, 1),
        'ave_i_frame_lpips': cur_ave_i_frame_lpips / max(cur_i_frame_num, 1),
        'frame_bpp': list(np.array(bits) / frame_pixel_num),
        'frame_psnr': psnrs,
        'frame_msssim': msssims,
        'frame_lpips': lpipses,
        'frame_type': frame_types,
        'test_time': test_time,
    }

    if cur_p_frame_num > 0:
        total_p_pixel_num = cur_p_frame_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = cur_ave_p_frame_bit / total_p_pixel_num
        log_result['ave_p_frame_psnr'] = cur_ave_p_frame_psnr / cur_p_frame_num
        log_result['ave_p_frame_msssim'] = cur_ave_p_frame_msssim / cur_p_frame_num
        log_result['ave_p_frame_lpips'] = cur_ave_p_frame_lpips / cur_p_frame_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_psnr'] = 0
        log_result['ave_p_frame_msssim'] = 0
        log_result['ave_p_frame_lpips'] = 0

    log_result['ave_all_frame_bpp'] = (cur_ave_i_frame_bit + cur_ave_p_frame_bit) / \
        (frame_num * frame_pixel_num)
    log_result['ave_all_frame_psnr'] = (cur_ave_i_frame_psnr + cur_ave_p_frame_psnr) / frame_num
    log_result['ave_all_frame_msssim'] = (cur_ave_i_frame_msssim + cur_ave_p_frame_msssim) / frame_num
    log_result['ave_all_frame_lpips'] = (cur_ave_i_frame_lpips + cur_ave_p_frame_lpips) / frame_num

    return log_result


def dump_json(obj, fid, float_digits=6, **kwargs):
    try:
        from unittest.mock import patch

        @patch('json.encoder.c_make_encoder', None)
        def _dump(obj, fid, float_digits, **kwargs):
            of = json.encoder._make_iterencode

            def inner(*args, **kwargs):
                args = list(args)
                args[4] = lambda o: format(o, '.%df' % float_digits)
                return of(*args, **kwargs)

            with patch('json.encoder._make_iterencode', wraps=inner):
                json.dump(obj, fid, **kwargs)

        _dump(obj, fid, float_digits, **kwargs)
    except Exception:
        # Fallback without custom float formatting
        json.dump(obj, fid, **kwargs)
