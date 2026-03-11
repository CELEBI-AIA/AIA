"""Synthetic benchmark for sequence-map creation used by DatasetLoader."""

import time
from typing import Any, Dict, List

from src.data_loader import _build_available_sequences_from_lists


def _legacy_loader_path_sequence_build(
    datasets_dir: str,
    all_images: List[str],
    all_videos: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Simulate previous loader path that needed an extra full traversal before map build."""
    rescanned_images: List[str] = []
    rescanned_videos: List[str] = []

    for path in [*all_images, *all_videos]:
        if path.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            rescanned_images.append(path)
        elif path.endswith((".mp4", ".avi", ".mov", ".mkv")):
            rescanned_videos.append(path)

    return _build_available_sequences_from_lists(datasets_dir, rescanned_images, rescanned_videos)


def _build_synthetic_paths(root: str, directories: int, frames_per_dir: int, videos: int):
    all_images: List[str] = []
    all_videos: List[str] = []

    for idx in range(directories):
        folder = f"{root}/mission_{idx:05d}"
        for frame in range(frames_per_dir):
            all_images.append(f"{folder}/frame_{frame:06d}.jpg")

    for idx in range(videos):
        all_videos.append(f"{root}/video_{idx:05d}.mp4")

    return all_images, all_videos


def test_sequence_builder_matches_legacy_behavior_and_is_faster():
    datasets_dir = "datasets"
    all_images, all_videos = _build_synthetic_paths(
        root=datasets_dir,
        directories=1800,
        frames_per_dir=12,
        videos=350,
    )

    expected = _legacy_loader_path_sequence_build(datasets_dir, all_images, all_videos)
    actual = _build_available_sequences_from_lists(datasets_dir, all_images, all_videos)
    assert actual == expected

    iterations = 25

    start_legacy = time.perf_counter()
    for _ in range(iterations):
        _legacy_loader_path_sequence_build(datasets_dir, all_images, all_videos)
    legacy_duration = time.perf_counter() - start_legacy

    start_new = time.perf_counter()
    for _ in range(iterations):
        _build_available_sequences_from_lists(datasets_dir, all_images, all_videos)
    new_duration = time.perf_counter() - start_new

    assert new_duration < legacy_duration
