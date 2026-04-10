"""
Annotate a RoboCasa LeRobot dataset with UVD subgoal segmentation.

For each episode, loads a camera video, runs UVD's embedding-based decomposition
to detect subgoal milestone frame indices, and writes results to:

    extras/episode_<id>/subgoal_segments.json

The JSON is keyed by "<backbone>_embed" so multiple backbones can be stored in
the same file. Re-running with a new backbone merges into the existing file
without touching previously computed keys (unless --overwrite is set).

Usage (single backbone):
    python annotate_uvd_subgoals.py \\
        --dataset_dir /path/to/lerobot/ \\
        --backbone dinov2

Usage (multiple backbones in one pass):
    python annotate_uvd_subgoals.py \\
        --dataset_dir /path/to/lerobot/ \\
        --backbone dinov2 vip clip \\
        --camera robot0_agentview_left \\
        --n_episodes 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Make UVD importable regardless of install state
_REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "UVD"))

from uvd.decomp import decomp_trajectories  # noqa: E402
from uvd.models import get_preprocessor  # noqa: E402

VALID_BACKBONES = ["vip", "r3m", "liv", "clip", "vc1", "dinov2", "resnet"]
VALID_CAMERAS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def _find_video(dataset_dir: Path, ep_num: int, camera: str) -> Path | None:
    matches = list(
        dataset_dir.glob(
            f"videos/*/observation.images.{camera}/episode_{ep_num:06d}.mp4"
        )
    )
    return matches[0] if matches else None


def annotate_dataset(
    dataset_dir: Path,
    backbones: list[str],
    camera: str,
    n_episodes: int | None,
    overwrite: bool,
    device: str,
) -> None:
    import decord

    dataset_dir = Path(dataset_dir)
    episode_dirs = sorted((dataset_dir / "extras").glob("episode_*"))
    if n_episodes is not None:
        episode_dirs = episode_dirs[:n_episodes]

    print(f"Dataset : {dataset_dir}")
    print(f"Episodes: {len(episode_dirs)}")
    print(f"Backbone: {backbones}")
    print(f"Camera  : {camera}")
    print(f"Device  : {device}")

    # Load all preprocessors once — avoids reloading model weights per episode
    preprocessors = {name: get_preprocessor(name, device=device) for name in backbones}

    for ep_dir in tqdm(episode_dirs, desc="episodes"):
        ep_num = int(ep_dir.name.split("_")[1])
        out_path = ep_dir / "subgoal_segments.json"

        # Merge with any existing results
        existing: dict = {}
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)

        to_run = [b for b in backbones if overwrite or f"{b}_embed" not in existing]
        if not to_run:
            continue

        video_path = _find_video(dataset_dir, ep_num, camera)
        if video_path is None:
            tqdm.write(
                f"[WARN] episode {ep_num}: no video for camera '{camera}', skipping."
            )
            continue

        vr = decord.VideoReader(str(video_path), height=224, width=224)
        frames = vr[:].asnumpy()  # (T, H, W, 3), uint8

        for backbone_name in to_run:
            embeddings = preprocessors[backbone_name].process(frames, return_numpy=True)
            _, decomp_meta = decomp_trajectories("embed", embeddings)
            existing[f"{backbone_name}_embed"] = {
                "milestone_indices": [int(i) for i in decomp_meta.milestone_indices],
            }

        with open(out_path, "w") as f:
            json.dump(existing, f)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to a LeRobot dataset directory (contains meta/, data/, videos/, extras/).",
    )
    parser.add_argument(
        "--backbone",
        nargs="+",
        default=["dinov2"],
        choices=VALID_BACKBONES,
        metavar="BACKBONE",
        help=f"Visual backbone(s) for UVD embedding. Choices: {VALID_BACKBONES}. "
        "Default: dinov2.",
    )
    parser.add_argument(
        "--camera",
        default="robot0_agentview_left",
        choices=VALID_CAMERAS,
        help="Camera view to use as input to UVD. Default: robot0_agentview_left.",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=None,
        help="Process only the first N episodes (useful for a quick test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite results even if they already exist in the JSON.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (default: cuda if available, else cpu).",
    )
    args = parser.parse_args()

    annotate_dataset(
        dataset_dir=Path(args.dataset_dir),
        backbones=args.backbone,
        camera=args.camera,
        n_episodes=args.n_episodes,
        overwrite=args.overwrite,
        device=args.device,
    )


if __name__ == "__main__":
    main()
