"""Aegis quadruped patrol demo.

Loads the bundled Aegis robot-dog URDF and renders a deterministic patrol
sequence: the dog trots forward, turns through two waypoints, and returns to a
stop. The motion is an open-loop kinematic showcase (not a learned policy), so
the result is fully reproducible from this single script.

Run from the repository root:

    python submissions/kocoodev-aegis-patrol/run_patrol_demo.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing demo dependency. Install with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc


# submissions/kocoodev-aegis-patrol/ -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_URDF = REPO_ROOT / "assets" / "Aegis" / "urdf" / "Aegis_mujoco.urdf"
DEFAULT_OUTPUT = HERE / "demo.mp4"
DEFAULT_TRAJECTORY = HERE / "trajectory.json"

LEGS = ("FL", "FR", "RR", "RL")
# Trot gait: diagonal pairs move together.
LEG_PHASE = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if value <= edge0:
        return 0.0
    if value >= edge1:
        return 1.0
    x = (value - edge0) / (edge1 - edge0)
    return x * x * (3.0 - 2.0 * x)


def ensure_mujoco_urdf(source_urdf: Path, output_urdf: Path) -> Path:
    """Add a <mujoco> compiler block so MuJoCo can load the SolidWorks URDF."""
    if output_urdf.exists():
        return output_urdf

    text = source_urdf.read_text(encoding="utf-8")
    text = re.sub(r'filename="\.\./meshes/([^"]+)"', r'filename="\1"', text)
    if "<mujoco>" not in text:
        text = text.replace(
            '<robot\n  name="Aegis">',
            '<robot\n  name="Aegis">\n  <mujoco>\n'
            '    <compiler meshdir="../meshes" discardvisual="false"/>\n'
            "  </mujoco>\n",
        )
    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    output_urdf.write_text(text, encoding="utf-8")
    return output_urdf


def build_model(urdf_path: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.option.timestep = 0.002
    spec.option.gravity = [0.0, 0.0, -9.81]

    base = spec.body("BASE_LINK")
    if base is None:
        raise ValueError("Missing BASE_LINK body in Aegis URDF")
    base.add_freejoint(name="floating_base_joint")

    world = spec.worldbody
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.06, 0.07, 0.09, 1.0],
    )
    # Patrol route markers so the path the dog follows reads clearly on camera.
    for idx, (px, py) in enumerate(((-1.1, 0.0), (0.4, 0.0), (1.2, 1.0))):
        world.add_geom(
            name=f"waypoint_{idx}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[px, py, 0.004],
            size=[0.13, 0.004, 0.0],
            rgba=[0.10, 0.45 + 0.15 * idx, 1.0, 0.55],
        )
    world.add_light(pos=[0.0, -1.4, 2.6], dir=[0, 0.4, -1], diffuse=[1.0, 1.0, 1.0])
    world.add_light(pos=[-1.4, 1.0, 1.8], dir=[0.4, -0.3, -1], diffuse=[0.5, 0.55, 0.65])
    return spec.compile()


def style_model_for_video(model: mujoco.MjModel) -> None:
    body_shell = np.array([0.92, 0.95, 1.00, 1.0], dtype=np.float32)
    hip_shell = np.array([1.00, 0.52, 0.12, 1.0], dtype=np.float32)
    leg_shell = np.array([0.18, 0.22, 0.28, 1.0], dtype=np.float32)
    foot_shell = np.array([0.05, 0.06, 0.07, 1.0], dtype=np.float32)

    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "floor" or name.startswith("waypoint_"):
            continue

        # Hide the duplicated collision group; keep the visual silhouette clean.
        if model.geom_group[geom_id] == 0:
            model.geom_rgba[geom_id] = [0.0, 0.0, 0.0, 0.0]
            continue

        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        ) or ""
        if body_name == "BASE_LINK":
            model.geom_rgba[geom_id] = body_shell
        elif "ABAD" in body_name or "HIP" in body_name:
            model.geom_rgba[geom_id] = hip_shell
        elif "FOOT" in body_name:
            model.geom_rgba[geom_id] = foot_shell
        else:
            model.geom_rgba[geom_id] = leg_shell


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    qpos_addr = int(model.jnt_qposadr[joint_id])
    if model.jnt_limited[joint_id]:
        low, high = model.jnt_range[joint_id]
        value = float(np.clip(value, low, high))
    data.qpos[qpos_addr] = value


def patrol_speed(time_s: float, duration_s: float) -> float:
    """Forward speed (m/s) along the heading, with a smooth start and stop."""
    ramp_in = smoothstep(0.0, 1.2, time_s)
    ramp_out = 1.0 - smoothstep(duration_s - 1.4, duration_s - 0.2, time_s)
    return 0.34 * ramp_in * ramp_out


def patrol_yaw_rate(time_s: float) -> float:
    """Turn rate (rad/s). Two timed turns shape the patrol route."""
    turn1 = 0.55 * (smoothstep(4.0, 4.6, time_s) - smoothstep(6.4, 7.0, time_s))
    turn2 = 0.50 * (smoothstep(9.5, 10.1, time_s) - smoothstep(12.0, 12.6, time_s))
    return turn1 + turn2


def apply_gait(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_x: float,
    base_y: float,
    yaw: float,
    gait_phase: float,
    moving: float,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    bob = 0.33 + moving * 0.02 * math.sin(2.0 * gait_phase)
    data.qpos[0] = base_x
    data.qpos[1] = base_y
    data.qpos[2] = bob
    data.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

    for leg in LEGS:
        phase = LEG_PHASE[leg]
        swing = math.sin(gait_phase + phase)
        lift = max(0.0, swing)
        set_joint(model, data, f"{leg}_ABAD_JOINT", moving * 0.08 * math.sin(gait_phase + phase + 0.4))
        set_joint(model, data, f"{leg}_HIP_JOINT", 0.58 + moving * 0.34 * swing)
        set_joint(model, data, f"{leg}_KNEE_JOINT", -1.08 + moving * 0.42 * lift)

    mujoco.mj_forward(model, data)


def update_camera(model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera, time_s: float) -> None:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BASE_LINK")
    lookat = data.xpos[base_id].copy() if base_id >= 0 else np.array([0.0, 0.0, 0.3])
    lookat[2] = 0.2

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = 1.9
    camera.azimuth = 120.0 + 35.0 * smoothstep(0.0, 15.0, time_s)
    camera.elevation = -16.0


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing body in model: {body_name}")
    return data.xpos[body_id].copy().round(5).tolist()


def run_demo(
    *,
    urdf_path: Path,
    video_path: Path,
    trajectory_path: Path,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
) -> dict:
    source_urdf = urdf_path
    if urdf_path.name == "Aegis_mujoco.urdf" and not urdf_path.exists():
        source_urdf = urdf_path.with_name("Aegis.urdf")
    if source_urdf.name == "Aegis.urdf":
        urdf_path = ensure_mujoco_urdf(source_urdf, urdf_path)

    model = build_model(urdf_path)
    style_model_for_video(model)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()

    video_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    trajectory: list[dict] = []
    total_frames = int(duration_s * fps)
    dt = 1.0 / fps

    base_x, base_y, yaw = -1.1, 0.0, 0.0
    gait_phase = 0.0
    gait_freq = 2.0 * math.pi * 1.7  # trot cadence

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps
        speed = patrol_speed(time_s, duration_s)
        moving = smoothstep(0.02, 0.12, speed)

        yaw += patrol_yaw_rate(time_s) * dt
        base_x += speed * math.cos(yaw) * dt
        base_y += speed * math.sin(yaw) * dt
        gait_phase += gait_freq * dt * (0.35 + 0.65 * moving)

        apply_gait(
            model,
            data,
            base_x=base_x,
            base_y=base_y,
            yaw=yaw,
            gait_phase=gait_phase,
            moving=moving,
        )
        update_camera(model, data, camera, time_s)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())

        if frame_idx % max(1, fps // 10) == 0:
            trajectory.append(
                {
                    "time_s": round(time_s, 3),
                    "base_pos": body_position(model, data, "BASE_LINK"),
                    "yaw": round(yaw, 4),
                    "speed": round(speed, 4),
                }
            )

    final_pos = body_position(model, data, "BASE_LINK")
    summary = {
        "project": "Aegis Quadruped Patrol",
        "task": (
            "The Aegis robot dog trots along a marked patrol route, turns through "
            "two waypoints, and comes to a controlled stop."
        ),
        "model": str(urdf_path),
        "source": str(REPO_ROOT / "assets" / "Aegis"),
        "video": str(video_path),
        "trajectory": str(trajectory_path),
        "duration_s": duration_s,
        "fps": fps,
        "distance_traveled_m": round(float(np.hypot(final_pos[0] + 1.1, final_pos[1])), 3),
        "final_base_pos": final_pos,
        "trajectory_samples": trajectory,
    }

    try:
        iio.imwrite(video_path, np.asarray(frames), fps=fps, codec="libx264")
    except Exception as exc:  # pragma: no cover - codec fallback
        fallback = video_path.with_suffix(".gif")
        iio.imwrite(fallback, np.asarray(frames), fps=fps)
        summary["video"] = str(fallback)
        summary["video_fallback_reason"] = str(exc)

    trajectory_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Aegis quadruped patrol demo video.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_demo(
        urdf_path=args.urdf,
        video_path=args.output,
        trajectory_path=args.trajectory,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "trajectory_samples"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
