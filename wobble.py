import os
import numpy as np
import cv2
from collections import deque
from numpy.lib.stride_tricks import sliding_window_view


def split_into_patches(frame, params):
    h, w, block_size, overlaps = params
    stride = block_size // 2 if overlaps else block_size
    return sliding_window_view(frame, (block_size, block_size))[::stride, ::stride]


def merge_patches(patches, params):
    h, w, block_size, overlaps = params

    h_block = h // block_size
    w_block = w // block_size
    stride = block_size // 2
    center_start = stride // 2
    center_end = center_start + stride

    patches = patches.transpose(0, 2, 1, 3)

    if overlaps:
        # merge rows
        rows = np.concatenate(
            [
                patches[0, :center_end],
                patches[1:-1, center_start:center_end].reshape(
                    (h - block_size - stride, patches.shape[2], block_size)
                ),
                patches[-1, center_start:],
            ],
            axis=0,
        )

        # merge cols
        result = np.concatenate(
            [
                rows[:, 0, :center_end],
                rows[:, 1:-1, center_start:center_end].reshape(
                    (h, w - block_size - stride)
                ),
                rows[:, -1, center_start:],
            ],
            axis=1,
        )

        return result
    else:
        return patches.reshape((h, w))


def process_frame(frame, params):
    h, w, block_size, overlaps = params
    patches = split_into_patches(frame, params)
    return np.fft.fft2(patches)


def unprocess_frame(frame, params):
    h, w, block_size, overlaps = params
    patches = np.fft.ifft2(frame)
    return merge_patches(patches, params)


def calculate_movement(average, frame, params):
    h, w, block_size, overlaps = params

    h_block = h // block_size
    w_block = w // block_size
    stride = block_size // 2

    div = np.angle(np.nan_to_num(frame / average))

    idx = np.r_[stride+6-block_size:0, 0:stride-5]
    x1, x2 = np.meshgrid(idx, idx, indexing="ij")
    X = np.column_stack([x1.ravel(), x2.ravel()])
    XtX_inv_Xt = np.linalg.inv(X.T @ X) @ X.T

    sub = div[:, :, idx][:, :, :, idx]
    y = sub.reshape(frame.shape[0], frame.shape[1], -1)

    coef = np.einsum("ij,abj->abi", XtX_inv_Xt, y)
    a, b = coef[...,0], coef[...,1]

    i = np.r_[0:stride+1, stride+1-block_size:0][:, None]
    j = np.r_[0:stride+1, stride+1-block_size:0][None, :]
    pred = a[..., None, None] * i + b[..., None, None] * j
    return pred


def move_frame(frame, movement, params):
    h, w, block_size, overlaps = params
    factor = np.exp(1j * movement)
    return frame * factor


def wobble(video, output, wobble_factor=30, block_size=30, overlaps=False):
    assert (
        block_size % 2 == 0 or not overlaps
    ), "block size must be even if overlaps is false"
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = h // block_size * block_size
    w = w // block_size * block_size
    params = (h, w, block_size, overlaps)
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    # getting an average
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    average_frame = np.zeros((h, w), dtype=np.uint32)
    actual_frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        actual_frame_count += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[:h, :w]

        average_frame += frame
    average_frame = average_frame / actual_frame_count

    # process average
    average_frame = process_frame(average_frame, params)

    # exaggerate movement and writing file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, frame_rate, (w, h), False)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(actual_frame_count):
        frame = video.read()[1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[:h, :w]

        frame = process_frame(frame, params)

        movement = calculate_movement(average_frame, frame, params)
        movement *= wobble_factor
        frame = move_frame(frame, movement, params)

        frame = unprocess_frame(frame, params)

        frame = frame.real.astype(np.uint8)
        out.write(frame)

    out.release()
    # for some reason opencv cannot immediately write with h264 codec on my machine.
    os.system(f'ffmpeg -i "{output}" -vcodec libx264 temp.mp4')
    os.system(f'mv temp.mp4 "{output}"')
