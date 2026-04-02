"""
Visually verify SubtaskContextWrapper by replaying a recorded episode and
rendering a video with the current subtask label burnt into each frame.

The video shows the camera feed with two text overlays:
  - Top:    original task instruction
  - Bottom: current subtask label (changes as the robot progresses)

This makes it easy to confirm that subtask transitions fire at the right moments.

Usage:
    python verify_subtask_wrapper.py \\
        --dataset_dir /path/to/datasets/.../prepare_coffee/20250812 \\
        --episode 0 \\
        --output_video /tmp/subtask_verify.mp4 \\
        --camera robot0_agentview_left

Also works without a dataset to do a quick smoke-test with random actions:
    python verify_subtask_wrapper.py --task PrepareCoffee --random_steps 200
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import imageio
import numpy as np

# ── optional: prefer cv2 for text rendering, fall back to PIL ────────────────
try:
    import cv2

    _RENDER_BACKEND = "cv2"
except ImportError:
    from PIL import Image, ImageDraw, ImageFont

    _RENDER_BACKEND = "pil"


# ---------------------------------------------------------------------------
# Text-overlay helpers
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int = 52) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _overlay_cv2(frame: np.ndarray, top_text: str, bottom_text: str) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.45, 1
    pad = 6

    def _put(text, y_start, color):
        for i, line in enumerate(text.splitlines()):
            y = y_start + i * 18
            # shadow
            cv2.putText(
                img,
                line,
                (pad + 1, y + 1),
                font,
                scale,
                (0, 0, 0),
                thickness + 1,
                cv2.LINE_AA,
            )
            cv2.putText(img, line, (pad, y), font, scale, color, thickness, cv2.LINE_AA)

    _put(_wrap(top_text), 18, (220, 220, 220))
    lines_bottom = _wrap(bottom_text).splitlines()
    y_bot = h - pad - len(lines_bottom) * 18
    _put(_wrap(bottom_text), y_bot, (80, 220, 80))
    return img


def _overlay_pil(frame: np.ndarray, top_text: str, bottom_text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    draw.text((6, 4), _wrap(top_text), fill=(220, 220, 220), font=font)
    h = img.height
    lines = _wrap(bottom_text).splitlines()
    y = h - 16 * len(lines) - 4
    draw.text((6, y), _wrap(bottom_text), fill=(80, 220, 80), font=font)
    return np.array(img)


def overlay_text(frame: np.ndarray, top_text: str, bottom_text: str) -> np.ndarray:
    if _RENDER_BACKEND == "cv2":
        return _overlay_cv2(frame, top_text, bottom_text)
    return _overlay_pil(frame, top_text, bottom_text)


# ---------------------------------------------------------------------------
# Replay from dataset
# ---------------------------------------------------------------------------


def verify_from_dataset(
    dataset_dir: Path,
    episode_idx: int,
    output_video: Path,
    camera: str,
    cam_height: int,
    cam_width: int,
):
    import robosuite
    import robocasa  # noqa: F401
    import robocasa.utils.lerobot_utils as LU
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.wrappers.subtask_context_wrapper import SUBTASK_REGISTRY

    dataset_dir = Path(dataset_dir)

    # Build env
    env_meta = LU.get_env_metadata(dataset_dir)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["camera_names"] = [camera]
    env_kwargs["camera_heights"] = cam_height
    env_kwargs["camera_widths"] = cam_width

    task_name = env_meta["env_name"]
    subtasks = SUBTASK_REGISTRY.get(task_name)
    print(f"Task: {task_name}")
    if subtasks is None:
        print(
            f"[WARN] '{task_name}' not in SUBTASK_REGISTRY – subtask labels will be empty."
        )

    env = robosuite.make(**env_kwargs)

    # Load episode data
    states = LU.get_episode_states(dataset_dir, episode_idx)
    model_xml = LU.get_episode_model_xml(dataset_dir, episode_idx)
    ep_meta = LU.get_episode_meta(dataset_dir, episode_idx)
    base_lang = ep_meta.get("lang", "")

    print(f"Episode {episode_idx}: {states.shape[0]} steps")
    print(f"Instruction: {base_lang}")

    # Full reset at episode start
    reset_to(
        env, {"model": model_xml, "ep_meta": json.dumps(ep_meta), "states": states[0]}
    )

    writer = imageio.get_writer(str(output_video), fps=20, codec="h264", quality=7)

    def _get_subtask_label(t: int) -> str:
        if subtasks is None:
            return "(no subtask registry entry)"
        for i, st in enumerate(subtasks[:-1]):
            try:
                if not st.is_complete(env):
                    n, total = i + 1, len(subtasks)
                    return f"[{n}/{total}] {st.description}"
            except Exception:
                return f"[?] detection error at step {t}"
        last = subtasks[-1]
        return f"[{len(subtasks)}/{len(subtasks)}] {last.description}"

    prev_label = ""
    for t in range(states.shape[0]):
        reset_to(env, {"states": states[t]})

        # Keep sticky flags up to date (e.g. GetToastedBread.toaster_on)
        if hasattr(env, "_check_success"):
            try:
                env._check_success()
            except Exception:
                pass

        label = _get_subtask_label(t)
        if label != prev_label:
            print(f"  step {t:4d}: {label}")
            prev_label = label

        # Render frame
        frame = env.sim.render(height=cam_height, width=cam_width, camera_name=camera)[
            ::-1
        ]
        frame = overlay_text(frame, base_lang, label)
        writer.append_data(frame)

    writer.close()
    env.close()
    print(f"\nVideo saved to: {output_video}")


# ---------------------------------------------------------------------------
# Smoke-test with random actions (no dataset needed)
# ---------------------------------------------------------------------------


def verify_random(
    task_name: str, n_steps: int, output_video: Path, cam_height: int, cam_width: int
):
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.wrappers.subtask_context_wrapper import (
        SubtaskContextWrapper,
        SUBTASK_REGISTRY,
    )

    subtasks = SUBTASK_REGISTRY.get(task_name)
    if subtasks is None:
        print(f"[WARN] '{task_name}' not in SUBTASK_REGISTRY.")

    env = gym.make(f"robocasa/{task_name}", split="target", enable_render=True)
    env = SubtaskContextWrapper(env)

    obs, _ = env.reset()
    base_lang = env._base_lang  # set during reset

    print(f"Task:        {task_name}")
    print(f"Instruction: {base_lang}")
    print(f"Running {n_steps} random steps …")

    # Access underlying robosuite env for rendering
    raw_env = env._get_kitchen_env()

    writer = imageio.get_writer(str(output_video), fps=20, codec="h264", quality=7)
    prev_label = ""

    for t in range(n_steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        full_lang = obs.get("annotation.human.task_description", "")
        # Extract the subtask suffix
        if "[Progress:" in full_lang:
            label = "[Progress:" + full_lang.split("[Progress:")[1]
        else:
            label = "(no subtask context)"

        if label != prev_label:
            print(f"  step {t:4d}: {label}")
            prev_label = label

        frame = raw_env.sim.render(
            height=cam_height, width=cam_width, camera_name="robot0_agentview_left"
        )[::-1]
        frame = overlay_text(frame, base_lang, label)
        writer.append_data(frame)

        if terminated or truncated:
            obs, _ = env.reset()
            break

    writer.close()
    env.close()
    print(f"\nVideo saved to: {output_video}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Dataset-replay mode
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="LeRobot dataset directory to replay from.",
    )
    parser.add_argument(
        "--episode", type=int, default=0, help="Episode index to replay (default: 0)."
    )

    # Random-action smoke-test mode
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task class name for random-action smoke test (e.g. PrepareCoffee).",
    )
    parser.add_argument(
        "--random_steps",
        type=int,
        default=300,
        help="Number of random-action steps (smoke-test mode).",
    )

    # Common
    parser.add_argument(
        "--output_video",
        type=str,
        default="/tmp/subtask_verify.mp4",
        help="Output video path (default: /tmp/subtask_verify.mp4).",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="robot0_agentview_left",
        help="Camera name to render (dataset-replay mode).",
    )
    parser.add_argument("--cam_height", type=int, default=256)
    parser.add_argument("--cam_width", type=int, default=256)

    args = parser.parse_args()

    if args.dataset_dir is not None:
        verify_from_dataset(
            dataset_dir=Path(args.dataset_dir),
            episode_idx=args.episode,
            output_video=Path(args.output_video),
            camera=args.camera,
            cam_height=args.cam_height,
            cam_width=args.cam_width,
        )
    elif args.task is not None:
        verify_random(
            task_name=args.task,
            n_steps=args.random_steps,
            output_video=Path(args.output_video),
            cam_height=args.cam_height,
            cam_width=args.cam_width,
        )
    else:
        parser.error("Provide --dataset_dir (replay mode) or --task (smoke-test mode).")


if __name__ == "__main__":
    main()
