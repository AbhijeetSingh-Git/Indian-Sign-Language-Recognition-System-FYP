"""
data_preprocessing.py
---------------------
Load sign language videos using OpenCV and extract a fixed number of frames.
Handles variable-length videos via uniform sampling + padding/truncation.
"""

import os
import cv2
import numpy as np


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SEQUENCE_LENGTH = 30          # fixed number of frames per video
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")


def extract_frames(video_path: str, num_frames: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Read a video file and return exactly `num_frames` RGB frames
    by sampling at evenly-spaced intervals.

    Parameters
    ----------
    video_path : str
        Absolute path to the .MP4 file.
    num_frames : int
        Number of frames to extract (default 30).

    Returns
    -------
    np.ndarray of shape (num_frames, H, W, 3) or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"[WARNING] Video has 0 frames: {video_path}")
        #cap.release()
        return None

    # Calculate evenly-spaced frame indices
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Fewer frames than needed → take all and pad later
        indices = np.arange(total_frames)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame to reduce  RAM usage                <--- changes 
            frame_rgb = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_rgb)
    cap.release()

    # Pad with black frames if video was shorter than num_frames
    if len(frames) < num_frames:
        h, w, c = frames[0].shape if frames else (224, 224, 3)
        while len(frames) < num_frames:
            frames.append(np.zeros((h, w, c), dtype=np.uint8))

    # Truncate to exact length (safety)
    frames = frames[:num_frames]
    return np.array(frames)


def load_dataset(video_dir: str = VIDEO_DIR, num_frames: int = SEQUENCE_LENGTH):
    """
    Walk through the video directory structure and load all videos.

    Expected structure:
        videos/
            class_name_1/
                video1.MP4
                video2.MP4
            class_name_2/
                ...

    Returns
    -------
    videos : list[np.ndarray]   – each element is (num_frames, H, W, 3)
    labels : list[str]          – corresponding class label strings
    class_names : list[str]     – sorted unique class names
    """
    videos = []
    labels = []

    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    class_names = sorted([
        d for d in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, d))
    ])

    print(f"[INFO] Found {len(class_names)} classes: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(video_dir, class_name)
        video_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]
        print(f"  -> {class_name}: {len(video_files)} videos")

        for vf in video_files:
            video_path = os.path.join(class_path, vf)
            frames = extract_frames(video_path, num_frames)
            if frames is not None:
                videos.append(frames)
                labels.append(class_name)

    print(f"[INFO] Loaded {len(videos)} videos total")
    return videos, labels, class_names


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    vids, lbls, classes = load_dataset()
    print(f"Classes: {classes}")
    print(f"Sample shape: {vids[0].shape}")
