# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Literal, Dict, Set

import cv2
import torch
import math
import os
import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from pathlib import Path
from PIL import Image

from scenedetect import open_video
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from torchvision import transforms
from torchvision.transforms import InterpolationMode


@lru_cache
def download_video_asset(filename: str) -> str:
    """
    Download and open an image from huggingface
    repo: raushan-testing-hf/videos-test
    """
    path = Path(os.path.join(os.path.expanduser("~/.cache/vllm"), "assets"))
    path.mkdir(parents=True, exist_ok=True)
    video_directory = path / "video-example-data"
    video_directory.mkdir(parents=True, exist_ok=True)

    video_path = video_directory / filename
    video_path_str = str(video_path)
    if not video_path.exists():
        video_path_str = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test",
            filename=filename,
            repo_type="dataset",
            cache_dir=video_directory,
        )
    return video_path_str


def video_to_ndarrays(path: str, num_frames: int = -1) -> npt.NDArray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total_frames = 0
    while cap.grab():
        total_frames += 1
    frames = []

    num_frames = num_frames if num_frames > 0 else total_frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    cap.release()
    cap = cv2.VideoCapture(path)
    for idx in range(total_frames):
        ok = cap.grab()  # next img
        if not ok:
            break
        if idx in frame_indices:  # only decompress needed
            ret, frame = cap.retrieve()
            if ret:
                frames.append(frame)

    frames = np.stack(frames)
    if len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video file {path}"
                         f" (expected {num_frames} frames, got {len(frames)})")
    return frames


def video_to_pil_images_list(path: str,
                             num_frames: int = -1) -> list[Image.Image]:
    frames = video_to_ndarrays(path, num_frames)
    return [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in frames
    ]


def build_cached_path(path: str, strategy: str, strategy_params: Dict[str, float] | None = None, num_frames: int = -1) -> str:
    PATH_TO_CACHE = "/srv/muse-lab/cache/"

    rel_path = path.split("videos" + os.sep)[-1]
    rel_dir = os.path.dirname(rel_path)
    filename = os.path.basename(rel_path)
    filename_no_ext = os.path.splitext(filename)[0]

    cache_dir = os.path.join(PATH_TO_CACHE, rel_dir)

    cached_filename = f"{filename_no_ext}_max_frames_{num_frames}_{strategy}"
    for i in strategy_params:
        num = str(strategy_params[i]).replace(".", "_")
        cached_filename += f"_{i}_{num}"
    cached_filename += ".npy"

    return os.path.join(cache_dir, cached_filename)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def calculate_resize_dims(height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 150800) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def smart_resize_frames(ret: npt.NDArray) -> npt.NDArray:
    MIN_PIXELS = 4 * 28 * 28
    MAX_PIXELS = 150800
    IMAGE_FACTOR = 28
    
    _, height, width, _ = ret.shape
    resized_height, resized_width = calculate_resize_dims(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    processed_frames = []
    for frame in ret:
        tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        resized = transforms.functional.resize(
            tensor_frame,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        processed_frames.append(resized.permute(1, 2, 0).byte().numpy())

    return np.stack(processed_frames)


def _thin_out_frames(selected_indices: list[int], max_frames: int) -> list[int]:
    """
    If a selection strategy yields too many frames, this function thins them out
    uniformly to a maximum number of frames. Ensures unique and sorted indices.

    Args:
        selected_indices (list[int]): A list of frame indices identified by a strategy.
        max_frames (int): The maximum number of frames desired in the output.

    Returns:
        list[int]: A new list of unique, sorted, and thinned frame indices.
    """
    unique_sorted_indices = sorted(list(set(selected_indices)))

    if len(unique_sorted_indices) <= max_frames:
        return unique_sorted_indices

    # Use linspace to select evenly spaced indices from the already selected ones
    # np.round is used to ensure we get integer indices from the float array
    thinning_indices_float = np.linspace(0, len(unique_sorted_indices) - 1, max_frames)
    thinning_indices = np.round(thinning_indices_float).astype(int)
    
    return [unique_sorted_indices[i] for i in thinning_indices]


def sample_frames_by_motion(video_path: str, max_frames: int = 32, motion_threshold: float = 1.0) -> npt.NDArray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    ret, prev_frame_bgr = cap.read()    # Read the first frame
    if not ret:
        cap.release()
        raise ValueError(f"Could not read first frame from {video_path}.")
    
    prev_frame_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    
    selected_indices = [0] # Always include the first frame as a starting point
    current_frame_idx = 1
    while True:
        ret, next_frame_bgr = cap.read()
        if not ret:
            break # End of video
        
        next_frame_gray = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate Farneback optical flow (dense flow)
        # Parameters for Farneback (tuned for general use, can be adjusted):
        # pyr_scale=0.5: pyramid scale, reduces image size by half at each level
        # levels=3: number of pyramid layers
        # winsize=15: averaging window size;
        # larger -> smoother motion, less noise, more computation
        # iterations=3: number of iterations at each pyramid level
        # poly_n=5, poly_sigma=1.2: polynomial expansion size & std for Gaussian
        # flags=0: typically 0 for standard optical flow
        # (or cv2.OPTFLOW_FARNEBACK_GAUSSIAN for Gaussian window)
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, 
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate the magnitude (length) of the flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate the average motion magnitude across the entire frame
        avg_motion = np.mean(magnitude)
        
        # Select frame if average motion exceeds the threshold
        if avg_motion > motion_threshold:
            selected_indices.append(current_frame_idx)
            
        prev_frame_gray = next_frame_gray # Update previous frame for next iter
        current_frame_idx += 1
        
    cap.release()

    # If the initial frame was selected, or no significant motion
    if len(selected_indices) <= 1:
        print(f"Warning: Few or no frames selected by motion for {video_path}. Falling back to uniform sampling.", flush=True)
        return video_to_ndarrays(video_path, max_frames)

    # Thin out selected frames if too many
    final_indices = _thin_out_frames(selected_indices, max_frames)
    
    # Read only the selected frames using OpenCV for efficiency
    sampled_frames = []
    cap = cv2.VideoCapture(video_path) # Reopen cap for specific frame access
    if not cap.isOpened():
        raise ValueError(f"Could not re-open video file {video_path} for reading sampled frames.")

    for idx in final_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
        else:
            print(f"Warning: Could not read frame at index {idx} from {video_path}. Skipping.", flush=True)
    cap.release()

    if not sampled_frames:
        raise ValueError(f"No frames were successfully sampled from {video_path} using motion detection.")

    return np.stack(sampled_frames)


def sample_frames_by_sharpness(video_path: str, max_frames: int = 32, sharpness_threshold: float = 100.0) -> npt.NDArray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    selected_indices = []
    current_frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # Convert to grayscale, as Laplacian works on single-channel images
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian operator
        # cv2.CV_64F is used as the depth of the output image to avoid overflow
        # when computing variance, as Laplacian can produce negative values.
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        
        # The var of the Laplacian indicates the amount of edges (sharpness).
        sharpness_score = laplacian.var()
        
        # Select frame if its sharpness score exceeds the threshold
        if sharpness_score > sharpness_threshold:
            selected_indices.append(current_frame_idx)
            
        current_frame_idx += 1
        
    cap.release()

    # If no sharp frames are found or threshold is too high
    if not selected_indices:
        print(f"Warning: No frames selected by sharpness for {video_path}. Falling back to uniform sampling.", flush=True)
        return video_to_ndarrays(video_path, max_frames)

    # Thin out selected frames if too many
    final_indices = _thin_out_frames(selected_indices, max_frames)
    
    # Read only the selected frames using OpenCV for efficiency
    sampled_frames = []
    cap = cv2.VideoCapture(video_path) # Reopen cap for specific frame access
    if not cap.isOpened():
        raise ValueError(f"Could not re-open video file {video_path} for reading sampled frames.")

    for idx in final_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
        else:
            print(f"Warning: Could not read frame at index {idx} from {video_path}. Skipping.", flush=True)
    cap.release()

    if not sampled_frames:
        raise ValueError(f"No frames were successfully sampled from {video_path} using sharpness detection.")

    return np.stack(sampled_frames)


def sample_frames_by_scene_change(video_path: str, max_frames: int = 32, threshold: float = 27.0) -> npt.NDArray:
    # ------------------------------------------------------------------
    # 1. Scene-detect quickly on down-scaled frames
    # ------------------------------------------------------------------
    video = open_video(video_path)
    scene_manager = SceneManager()
    # ContentDetector: detects fast cuts using weighted average of HSV change.
    # The ContentDetector works by comparing successive frames of a video.
    # If difference <= threshold, the frames are considered part of the same
    # scene. Higher threshold: Fewer scene changes will be detected.
    # The detector will be less sensitive to minor changes and will only mark
    # very distinct cuts
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()  # (start_tc, end_tc) tuples
    # ------------------------------------------------------------------
    # 2. Pick candidate indices: first / mid / last of each scene
    # ------------------------------------------------------------------
    idx: Set[int] = set()
    for start_tc, end_tc in scene_list:
        start, end = start_tc.get_frames(), end_tc.get_frames()
        idx.add(start)
        if end - start > 3:
            idx.add((start + end) // 2)
        if end - start >= 1:
            idx.add(end - 1)
    # ------------------------------------------------------------------
    # 3. Thin if necessary – uniformly across the *already* selected idx
    # ------------------------------------------------------------------
    if len(idx) > max_frames:
        sorted_idx = sorted(idx)
        keep = np.linspace(0, len(sorted_idx) - 1, max_frames, dtype=int)
        idx = {sorted_idx[i] for i in keep}
    # ------------------------------------------------------------------
    # 4. Fallback: no scenes (or empty idx) → uniform sampling
    # ------------------------------------------------------------------
    # Maybe can be improved
    if not idx:
        # total = int(video.props.num_frames)
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        idx = set(np.linspace(0, total - 1, max_frames, dtype=int))

    sorted_idx: List[int] = sorted(idx)
    # ------------------------------------------------------------------
    # 5. Grab frames efficiently (single forward scan rather than random seeks)
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    next_target = 0
    target_count = len(sorted_idx)

    for frame_no in range(sorted_idx[-1] + 1):          # iterate once
        ok, frame = cap.read()
        if not ok:
            break                                       # EOF / corruption
        if frame_no == sorted_idx[next_target]:
            frames.append(frame[..., ::-1])             # BGR→RGB if needed
            next_target += 1
            if next_target == target_count:             # collected all
                break
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from {video_path}")
    
    return np.stack(frames)


def video_to_ndarrays_clever_sample(path: str, strategy: str, strategy_params: Dict[str, float] | None = None, num_frames: int = -1, smart_resize: bool = False) -> npt.NDArray:
    cached_path = build_cached_path(path, strategy, strategy_params, num_frames)

    if os.path.exists(cached_path):
        print(f"Found cached sampled frames at {cached_path}, loading...", flush=True)
        ret = np.load(cached_path)
    else:
        try:
            if strategy == "motion_based":
                motion_threshold = strategy_params.get("motion_threshold", 1.0)
                ret = sample_frames_by_motion(
                    path,
                    max_frames=num_frames,
                    motion_threshold=motion_threshold
                )
            elif strategy == "sharpness_based":
                sharpness_threshold = strategy_params.get("sharpness_threshold", 100.0)
                ret = sample_frames_by_sharpness(
                    path,
                    max_frames=num_frames,
                    sharpness_threshold=sharpness_threshold
                )
            elif strategy == "scene_change":
                content_threshold = strategy_params.get("content_threshold", 27.0)
                ret = sample_frames_by_scene_change(
                    path,
                    max_frames=num_frames,
                    threshold=content_threshold
                )
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
            
            np.save(cached_path, ret)
    
        except ValueError as e:
            print(f"Error during video processing for strategy '{strategy}': {e}", flush=True)
            return np.array([])
        except Exception as e:
            print(f"An unexpected error occurred during strategy '{strategy}': {e}", flush=True)
            return np.array([])
        
    if smart_resize:
        return smart_resize_frames(ret)

    return ret


def video_to_pil_images_list_clever_sample(path: str, strategy: str, strategy_params: Dict[str, float] | None = None, num_frames: int = -1, smart_resize: bool = False) -> list[Image.Image]:
    frames = video_to_ndarrays_clever_sample(
        path,
        strategy=strategy,
        strategy_params=strategy_params,
        num_frames=num_frames,
        smart_resize=smart_resize
    )
    return [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in frames
    ]


@dataclass(frozen=True)
class VideoAsset:
    name: Literal["sample_demo_1.mp4"]
    num_frames: int = -1
    strategy: str = "uniform"
    strategy_params: Dict[str, float] = field(default_factory=dict)
    smart_resize: bool = False

    @property
    def pil_images(self) -> list[Image.Image]:

        if self.strategy == "uniform":
            video_path = download_video_asset(self.name)
            ret = video_to_pil_images_list(video_path, self.num_frames)

            # TODO: Add smart_resize for pil images
            # if self.smart_resize:
            #     return smart_resize_frames(ret)
            return ret
        else:
            video_path = download_video_asset(self.name)
            ret = video_to_pil_images_list_clever_sample(
                video_path,
                strategy=self.strategy,
                strategy_params=self.strategy_params,
                num_frames=self.num_frames,
                smart_resize=self.smart_resize
            )
            return ret

    @property
    def np_ndarrays(self) -> npt.NDArray:
        if self.strategy == "uniform":
            video_path = download_video_asset(self.name)
            ret = video_to_ndarrays(video_path, self.num_frames)
            if self.smart_resize:
                return smart_resize_frames(ret)
            return ret
        else:
            video_path = download_video_asset(self.name)

            ret = video_to_ndarrays_clever_sample(
                video_path,
                strategy=self.strategy,
                strategy_params=self.strategy_params,
                num_frames=self.num_frames,
                smart_resize=self.smart_resize
            )
            return ret
