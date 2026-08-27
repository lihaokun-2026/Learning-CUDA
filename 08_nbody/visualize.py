import argparse
import struct
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
    return data.reshape(records, particles, 3)


def animate_2d(trajectory, output=None, interval=40, trail=0):
    fig, ax = plt.subplots(figsize=(8, 8))
    xy = trajectory[:, :, :2]
    limit = max(float(np.abs(xy).max()), 1.0) * 1.1
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x", ylabel="y", aspect="equal")
    points, = ax.plot([], [], "o", ms=3, alpha=0.8)
    title = ax.set_title("")

    def update(frame):
        start = max(0, frame - trail) if trail else frame
        visible = xy[start:frame + 1]
        points.set_data(visible[-1, :, 0], visible[-1, :, 1])
        title.set_text(f"N-body simulation — frame {frame + 1}/{len(xy)}")
        return points, title

    animation = FuncAnimation(fig, update, frames=len(xy), interval=interval, blit=True)
    if output:
        suffix = Path(output).suffix.lower()
        writer = "pillow" if suffix == ".gif" else "ffmpeg"
        animation.save(output, writer=writer, dpi=120)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize nbody trajectory.bin")
    parser.add_argument("trajectory", nargs="?", default="trajectory.bin")
    parser.add_argument("--output", help="GIF or MP4 output path")
    parser.add_argument("--interval", type=int, default=40)
    parser.add_argument("--trail", type=int, default=0, help="reserved for future trail rendering")
    args = parser.parse_args()
    animate_2d(load_trajectory(args.trajectory), args.output, args.interval, args.trail)


if __name__ == "__main__":
    main()
