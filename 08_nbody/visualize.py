import argparse
import struct
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def load_trajectory(path: str):
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError("trajectory file is too short")
        particles, records = struct.unpack("<ii", header)
        data = np.fromfile(f, dtype="<f4")
    expected = records * particles * 3
    if particles <= 0 or records <= 0 or data.size != expected:
        raise ValueError(f"invalid trajectory dimensions: particles={particles}, records={records}")
    # File layout is particle-major: [particle, record, (x, y, z)].
    # Animation code uses frame-major: [record, particle, (x, y, z)].
    return data.reshape(particles, records, 3).transpose(1, 0, 2)


def axis_limits(values):
    flattened = values.reshape(-1, values.shape[-1])
    minimum = np.percentile(flattened, 1.0, axis=0)
    maximum = np.percentile(flattened, 99.0, axis=0)
    # Include the full initial state when it is not an outlier.
    minimum = np.minimum(minimum, values[0].min(axis=0))
    maximum = np.maximum(maximum, values[0].max(axis=0))
    center = (minimum + maximum) * 0.5
    half_range = max(float((maximum - minimum).max()) * 0.55, 1.0e-3)
    return [(c - half_range, c + half_range) for c in center]


def save_or_show(animation, output):
    if output:
        suffix = Path(output).suffix.lower()
        writer = "pillow" if suffix == ".gif" else "ffmpeg"
        animation.save(output, writer=writer, dpi=120)
    else:
        plt.show()


def animate_2d(trajectory, output=None, interval=100, trail=0):
    fig, ax = plt.subplots(figsize=(8, 8))
    limits = axis_limits(trajectory[:, :, :2])
    ax.set(xlim=limits[0], ylim=limits[1], xlabel="x", ylabel="y", aspect="equal")
    points, = ax.plot([], [], "o", ms=4, alpha=0.85)
    trails = [ax.plot([], [], "-", lw=0.7, alpha=0.35)[0] for _ in range(trajectory.shape[1])] if trail else []
    title = ax.set_title("")

    def update(frame):
        start = max(0, frame - trail) if trail else frame
        visible = trajectory[start:frame + 1]
        points.set_data(visible[-1, :, 0], visible[-1, :, 1])
        for particle, line in enumerate(trails):
            line.set_data(visible[:, particle, 0], visible[:, particle, 1])
        title.set_text(f"N-body 2D — frame {frame + 1}/{len(trajectory)}")
        return (points, title, *trails)

    animation = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=True)
    save_or_show(animation, output)


def animate_3d(trajectory, output=None, interval=100, trail=0):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    limits = axis_limits(trajectory)
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    points = ax.scatter([], [], [], s=14, alpha=0.85)
    trails = [ax.plot([], [], [], "-", lw=0.7, alpha=0.35)[0] for _ in range(trajectory.shape[1])] if trail else []
    title = ax.set_title("")

    def update(frame):
        start = max(0, frame - trail) if trail else frame
        visible = trajectory[start:frame + 1]
        points._offsets3d = (visible[-1, :, 0], visible[-1, :, 1], visible[-1, :, 2])
        for particle, line in enumerate(trails):
            line.set_data_3d(visible[:, particle, 0], visible[:, particle, 1], visible[:, particle, 2])
        ax.view_init(elev=25, azim=35 + frame * 0.4)
        title.set_text(f"N-body 3D — frame {frame + 1}/{len(trajectory)}")
        return (points, title, *trails)

    animation = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False)
    save_or_show(animation, output)


def main():
    parser = argparse.ArgumentParser(description="Visualize nbody trajectory.bin")
    parser.add_argument("trajectory", nargs="?", default="trajectory.bin")
    parser.add_argument("--output", help="GIF or MP4 output path")
    parser.add_argument("--time-log", help="visualization timing log path")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="GIF/动画播放速度，默认约 10 帧/秒")
    parser.add_argument("--interval", type=int, default=None,
                        help="兼容参数：直接指定每帧毫秒数，会覆盖 --fps")
    parser.add_argument("--trail", type=int, default=20, help="number of previous frames in each trail; 0 disables trails")
    parser.add_argument("--dimension", choices=("2d", "3d"), default="3d")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    interval = args.interval if args.interval is not None else max(1, round(1000.0 / args.fps))
    total_start = time.perf_counter()
    trajectory = load_trajectory(args.trajectory)
    animate = animate_3d if args.dimension == "3d" else animate_2d
    animate(trajectory, args.output, interval, args.trail)
    elapsed = time.perf_counter() - total_start
    if args.output:
        log_path = Path(args.time_log) if args.time_log else Path(args.output).with_suffix(".visualization.log")
        output_size = Path(args.output).stat().st_size if Path(args.output).exists() else 0
        log_path.write_text(
            "\n".join((
                f"trajectory={args.trajectory}",
                f"output={args.output}",
                f"dimension={args.dimension}",
                f"frames={len(trajectory)}",
                f"particles={trajectory.shape[1]}",
                f"requested_fps={args.fps}",
                f"interval_ms={interval}",
                f"output_size_bytes={output_size}",
                f"visualization_total_sec={elapsed:.6f}",
            )) + "\n",
            encoding="utf-8",
        )
        print(f"Visualization time: {elapsed:.3f} s")
        print(f"Timing log: {log_path}")


if __name__ == "__main__":
    main()
