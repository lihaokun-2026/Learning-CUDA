import argparse
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np


OBJECT_STYLE = {
    "black_hole": ((255, 245, 120), 8),
    "neutron_star_white_dwarf": ((120, 220, 255), 4),
    "star": ((255, 170, 70), 3),
    "planet": ((80, 220, 130), 2),
    "asteroid": ((185, 195, 215), 1),
}


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


def axis_limits(values, z_scale=1.0):
    values = values.copy()
    if values.shape[-1] == 3:
        values[..., 2] *= z_scale
    flattened = values.reshape(-1, values.shape[-1])
    minimum = np.percentile(flattened, 1.0, axis=0)
    maximum = np.percentile(flattened, 99.0, axis=0)
    # Include the full initial state when it is not an outlier.
    minimum = np.minimum(minimum, values[0].min(axis=0))
    maximum = np.maximum(maximum, values[0].max(axis=0))
    center = (minimum + maximum) * 0.5
    half_range = max(float((maximum - minimum).max()) * 0.55, 1.0e-3)
    return [(c - half_range, c + half_range) for c in center]


def save_or_show(animation, output, dpi=120):
    if output:
        suffix = Path(output).suffix.lower()
        writer = "pillow" if suffix == ".gif" else "ffmpeg"
        animation.save(output, writer=writer, dpi=dpi)
    else:
        import matplotlib.pyplot as plt
        plt.show()


def animate_2d(trajectory, output=None, interval=100, trail=0, dpi=120):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

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
    save_or_show(animation, output, dpi)


def animate_3d(trajectory, object_types=None, output=None, interval=100, trail=0,
               dpi=120, z_scale=1.0):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    particles = trajectory.shape[1]
    if object_types is None:
        object_types = np.full(particles, "asteroid", dtype=object)
    draw_limit = 14000 if particles > 20000 else particles
    selected_groups = []
    for kind in OBJECT_STYLE:
        indices = np.flatnonzero(object_types == kind)
        if kind == "asteroid" and len(indices) > draw_limit:
            indices = indices[np.linspace(0, len(indices) - 1, draw_limit, dtype=int)]
        selected_groups.append(indices)
    selected = np.unique(np.concatenate(selected_groups))
    display = trajectory[:, selected].astype(np.float32, copy=True)
    display[:, :, 2] *= z_scale

    limit_frames = np.linspace(0, len(display) - 1, min(len(display), 101), dtype=int)
    limit_particles = np.linspace(0, len(selected) - 1,
                                  min(len(selected), 20000), dtype=int)
    limits = axis_limits(display[np.ix_(limit_frames, limit_particles)])

    fig = plt.figure(figsize=(11, 9), facecolor="#10131a")
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel="x", ylabel="y",
           zlabel=f"z (visual × {z_scale:g})")
    ax.set_box_aspect((1, 1, 0.85))
    ax.view_init(elev=28, azim=38)
    ax.set_facecolor("#10131a")
    ax.grid(True, alpha=0.22)
    title = ax.set_title("", color="white", pad=18, fontsize=13)

    selected_types = object_types[selected]
    scatters = {}
    for kind, (color, size) in OBJECT_STYLE.items():
        local = np.flatnonzero(selected_types == kind)
        if len(local):
            rgb = tuple(component / 255.0 for component in color)
            scatters[kind] = ax.scatter(
                display[0, local, 0], display[0, local, 1], display[0, local, 2],
                s=max(2.0, size * size * 0.75), c=[rgb],
                alpha=0.85 if kind != "asteroid" else 0.38,
                depthshade=True, edgecolors="none")

    # Batch all trail segments in one 3D collection. This shows substantially
    # more particle trajectories without creating one expensive Line3D object
    # per particle. Every body is still simulated and stored in the BIN file.
    trail_count = min(1024 if particles <= 20000 else 512, len(selected)) if trail else 0
    trail_indices = np.linspace(0, len(selected) - 1, trail_count, dtype=int)
    trail_collection = None
    if trail_indices.size:
        trail_collection = Line3DCollection([], colors="#8ab4f8", linewidths=0.65,
                                            alpha=0.58)
        ax.add_collection3d(trail_collection)
    legend_items = [
        Line2D([0], [0], marker="o", color="none", label=kind.replace("_", " "),
               markerfacecolor=np.array(color) / 255.0, markersize=max(4, size + 1))
        for kind, (color, size) in OBJECT_STYLE.items() if kind in scatters
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8,
              facecolor="#202631", labelcolor="white", framealpha=0.85)

    def update(frame):
        start = max(0, frame - trail) if trail else frame
        for kind, points in scatters.items():
            local = np.flatnonzero(selected_types == kind)
            current = display[frame, local]
            points._offsets3d = (current[:, 0], current[:, 1], current[:, 2])
        if trail_collection is not None:
            history = display[start:frame + 1, trail_indices]
            if len(history) > 1:
                segments = np.stack((history[:-1], history[1:]), axis=2)
                trail_collection.set_segments(segments.reshape(-1, 2, 3))
            else:
                trail_collection.set_segments([])
        title.set_text(f"N-body 3D · frame {frame + 1}/{len(trajectory)} · "
                       f"显示 {len(selected):,}/{particles:,} 个天体 · "
                       f"轨迹 {len(trail_indices):,} 条")
        animated = (*scatters.values(), title)
        return animated + ((trail_collection,) if trail_collection is not None else ())

    animation = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False)
    save_or_show(animation, output, dpi)
    plt.close(fig)


def find_objects_file(trajectory_path, explicit_path=None):
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"objects metadata not found: {path}")
        return path
    stem = Path(trajectory_path).stem
    candidates = (
        Path(trajectory_path).with_name(f"{stem}_objects.txt"),
        Path("data") / f"{stem}_objects.txt",
        Path(__file__).parent / "data" / f"{stem}_objects.txt",
    )
    return next((path for path in candidates if path.exists()), None)


def load_object_types(path, particles):
    names = np.full(particles, "asteroid", dtype=object)
    if path is None:
        return names
    rows = Path(path).read_text(encoding="utf-8").splitlines()[1:]
    if len(rows) != particles:
        raise ValueError(f"metadata has {len(rows)} objects, trajectory has {particles}")
    for row in rows:
        index, object_type, _mass = row.split()
        if object_type not in OBJECT_STYLE:
            raise ValueError(f"unknown object type: {object_type}")
        names[int(index)] = object_type
    return names


def projected_limits(trajectory, z_scale, azimuth=35.0, elevation=25.0):
    """Estimate fixed camera limits from a bounded sample without flattening 0.8 GiB."""
    frame_stride = max(1, trajectory.shape[0] // 101)
    particle_stride = max(1, trajectory.shape[1] // 20000)
    sample = trajectory[::frame_stride, ::particle_stride].reshape(-1, 3).astype(np.float64)
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    horizontal = np.cos(azimuth) * sample[:, 0] - np.sin(azimuth) * sample[:, 1]
    depth = np.sin(azimuth) * sample[:, 0] + np.cos(azimuth) * sample[:, 1]
    vertical = -np.sin(elevation) * depth + np.cos(elevation) * sample[:, 2] * z_scale
    low = np.percentile(np.column_stack((horizontal, vertical)), 0.2, axis=0)
    high = np.percentile(np.column_stack((horizontal, vertical)), 99.8, axis=0)
    center = (low + high) * 0.5
    half = np.maximum((high - low) * 0.55, 1.0e-6)
    return center, half


def ffmpeg_has_encoder(name):
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0 and name in result.stdout


def select_video_encoder(requested, gpu_name):
    """Select a portable H.264 encoder; A100/A800 GPUs do not contain NVENC."""
    if requested == "cpu":
        return "libx264"
    nvenc_unavailable_gpu = any(name in gpu_name.upper() for name in ("A100", "A800"))
    nvenc_available = ffmpeg_has_encoder("h264_nvenc") and not nvenc_unavailable_gpu
    if requested == "nvenc" and not nvenc_available:
        raise RuntimeError(
            f"h264_nvenc is unavailable on {gpu_name}; use --video-encoder cpu. "
            "A100/A800 supports CUDA rendering but has no NVENC hardware."
        )
    return "h264_nvenc" if nvenc_available else "libx264"


def render_gpu_video(trajectory, object_types, output, fps, trail, width, height,
                     z_scale, frame_stride=1, video_encoder="auto", device="cuda"):
    """Rasterize particles on CUDA and stream RGB frames to FFmpeg."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GPU backend requires PyTorch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA GPU")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("GPU backend requires ffmpeg in PATH")

    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    gpu_trajectory = torch.from_numpy(np.ascontiguousarray(trajectory)).to(device)
    center, half = projected_limits(trajectory, z_scale)
    center = torch.tensor(center, dtype=torch.float32, device=device)
    half = torch.tensor(half, dtype=torch.float32, device=device)
    azimuth = np.deg2rad(35.0)
    elevation = np.deg2rad(25.0)
    masks = {
        kind: torch.from_numpy(np.flatnonzero(object_types == kind)).to(device)
        for kind in OBJECT_STYLE
    }
    suffix = Path(output).suffix.lower()
    if suffix == ".mp4":
        selected_encoder = select_video_encoder(video_encoder, gpu_name)
        encoding_options = (
            ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
             "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            if selected_encoder == "h264_nvenc" else
            ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        )
    elif suffix == ".gif":
        selected_encoder = "gif"
        encoding_options = ["-c:v", "gif", "-gifflags", "+transdiff", "-loop", "0"]
    else:
        raise ValueError("GPU backend output must end with .mp4 or .gif")
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", *encoding_options, output,
    ]
    print(f"GPU: {gpu_name}")
    print(f"Video encoder: {selected_encoder}")
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    if encoder.stdin is None:
        raise RuntimeError("failed to open FFmpeg input pipe")

    render_seconds = 0.0
    encode_start = time.perf_counter()

    def project(points):
        horizontal = np.cos(azimuth) * points[:, 0] - np.sin(azimuth) * points[:, 1]
        depth = np.sin(azimuth) * points[:, 0] + np.cos(azimuth) * points[:, 1]
        vertical = -np.sin(elevation) * depth + np.cos(elevation) * points[:, 2] * z_scale
        x = ((horizontal - center[0]) / (2.0 * half[0]) + 0.5) * (width - 1)
        y = (0.5 - (vertical - center[1]) / (2.0 * half[1])) * (height - 1)
        return x.round().long(), y.round().long()

    def draw(canvas, x, y, color, radius, strength=1.0):
        radius = max(1, radius)
        for dy in range(-radius + 1, radius):
            extent = int(np.sqrt(max(0, radius * radius - dy * dy)))
            for dx in range(-extent + 1, extent):
                xx, yy = x + dx, y + dy
                valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                indices = yy[valid] * width + xx[valid]
                for channel, value in enumerate(color):
                    canvas[channel].view(-1).index_add_(
                        0, indices, torch.full_like(indices, value * strength, dtype=torch.float32)
                    )

    try:
        output_frames = list(range(0, len(trajectory), frame_stride))
        if output_frames[-1] != len(trajectory) - 1:
            output_frames.append(len(trajectory) - 1)
        progress_start = time.perf_counter()
        for output_index, frame in enumerate(output_frames, 1):
            frame_start = time.perf_counter()
            canvas = torch.zeros((3, height, width), dtype=torch.float32, device=device)
            if trail:
                first = max(0, frame - trail)
                history = gpu_trajectory[first:frame: max(1, trail // 5 or 1)].reshape(-1, 3)
                if len(history):
                    hx, hy = project(history)
                    draw(canvas, hx, hy, (40, 55, 80), 1, 0.18)
            points = gpu_trajectory[frame]
            x, y = project(points)
            for kind, (color, radius) in OBJECT_STYLE.items():
                indices = masks[kind]
                if len(indices):
                    draw(canvas, x[indices], y[indices], color, radius)
            canvas = canvas.clamp_(0, 255).permute(1, 2, 0).byte()
            frame_rgb = canvas.cpu().numpy()
            render_seconds += time.perf_counter() - frame_start
            encoder.stdin.write(frame_rgb.tobytes())
            if output_index == 1 or output_index % 25 == 0 or output_index == len(output_frames):
                progress_elapsed = time.perf_counter() - progress_start
                remaining = progress_elapsed / output_index * (len(output_frames) - output_index)
                print(
                    f"Rendered {output_index}/{len(output_frames)} frames "
                    f"({100.0 * output_index / len(output_frames):.1f}%), ETA {remaining:.1f} s",
                    flush=True,
                )
    finally:
        encoder.stdin.close()
    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg exited with status {return_code}")
    total_stream_seconds = time.perf_counter() - encode_start
    del gpu_trajectory
    torch.cuda.empty_cache()
    return render_seconds, max(0.0, total_stream_seconds - render_seconds), selected_encoder, gpu_name


def main():
    parser = argparse.ArgumentParser(description="Visualize nbody trajectory.bin")
    parser.add_argument("trajectory", nargs="?", default="trajectory.bin")
    parser.add_argument("--output", help="GIF or MP4 output path")
    parser.add_argument("--time-log", help="visualization timing log path")
    parser.add_argument("--objects", help="object metadata; benchmark files are detected automatically")
    parser.add_argument("--backend", choices=("auto", "gpu", "matplotlib"), default="auto",
                        help="auto uses CUDA+FFmpeg for MP4/GIF when available")
    parser.add_argument("--video-encoder", choices=("auto", "nvenc", "cpu"), default="auto",
                        help="MP4 encoder: NVENC when supported, otherwise portable libx264")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="GIF/动画播放速度，默认约 10 帧/秒")
    parser.add_argument("--interval", type=int, default=None,
                        help="兼容参数：直接指定每帧毫秒数，会覆盖 --fps")
    parser.add_argument("--trail", type=int, default=80, help="number of previous frames in each 3D trail; 0 disables trails")
    parser.add_argument("--dimension", choices=("2d", "3d"), default="3d")
    parser.add_argument("--z-scale", type=float, default=35.0,
                        help="visual-only vertical exaggeration for fixed 3D projection")
    parser.add_argument("--width", type=int, help="output width; auto: 1920 for 4096, 2560 for 65536")
    parser.add_argument("--height", type=int, help="output height; auto: 1080 for 4096, 1440 for 65536")
    parser.add_argument("--dpi", type=int, default=160, help="Matplotlib backend DPI")
    parser.add_argument("--frame-stride", type=int,
                        help="render every Nth trajectory frame; auto: 2 for 4096 and 4 for 65536")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    interval = args.interval if args.interval is not None else max(1, round(1000.0 / args.fps))
    total_start = time.perf_counter()
    trajectory = load_trajectory(args.trajectory)
    objects_path = find_objects_file(args.trajectory, args.objects)
    object_types = load_object_types(objects_path, trajectory.shape[1])
    width = args.width or (2560 if trajectory.shape[1] >= 65536 else 1920)
    height = args.height or (1440 if trajectory.shape[1] >= 65536 else 1080)
    output_suffix = Path(args.output).suffix.lower() if args.output else ""
    default_stride = 1 if output_suffix == ".mp4" else (4 if trajectory.shape[1] >= 65536 else 2)
    frame_stride = args.frame_stride or default_stride
    if frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    output_frames = (len(trajectory) - 1) // frame_stride + 1
    if (len(trajectory) - 1) % frame_stride:
        output_frames += 1
    backend = args.backend
    if backend == "auto":
        # Matplotlib is the default because it is the required, inspectable
        # FuncAnimation 3D implementation. Use --backend gpu explicitly for
        # the high-throughput rasterized video path.
        backend = "matplotlib"
    gpu_render_sec = 0.0
    encode_sec = 0.0
    selected_encoder = "matplotlib"
    gpu_name = "none"
    if backend == "gpu":
        if not args.output or Path(args.output).suffix.lower() not in (".mp4", ".gif"):
            parser.error("GPU backend requires a .mp4 or .gif output")
        gpu_render_sec, encode_sec, selected_encoder, gpu_name = render_gpu_video(
            trajectory, object_types, args.output, args.fps, args.trail,
            width, height, args.z_scale, frame_stride, args.video_encoder,
        )
    else:
        animate = animate_3d if args.dimension == "3d" else animate_2d
        if args.dimension == "3d":
            animate(trajectory, object_types, args.output, interval, args.trail,
                    args.dpi, args.z_scale)
        else:
            animate(trajectory, args.output, interval, args.trail, args.dpi)
    elapsed = time.perf_counter() - total_start
    if args.output:
        log_path = Path(args.time_log) if args.time_log else Path(args.output).with_suffix(".visualization.log")
        output_size = Path(args.output).stat().st_size if Path(args.output).exists() else 0
        log_path.write_text(
            "\n".join((
                f"trajectory={args.trajectory}",
                f"output={args.output}",
                f"dimension={args.dimension}",
                f"backend={backend}",
                f"gpu_name={gpu_name}",
                f"video_encoder={selected_encoder}",
                f"trajectory_frames={len(trajectory)}",
                f"output_frames={output_frames if backend == 'gpu' else len(trajectory)}",
                f"frame_stride={frame_stride if backend == 'gpu' else 1}",
                f"particles={trajectory.shape[1]}",
                f"objects_metadata={objects_path or 'none'}",
                f"requested_fps={args.fps}",
                f"interval_ms={interval}",
                f"width={width}",
                f"height={height}",
                f"z_scale={args.z_scale}",
                f"output_size_bytes={output_size}",
                f"gpu_render_and_transfer_sec={gpu_render_sec:.6f}",
                f"video_encoding_and_pipe_wait_sec={encode_sec:.6f}",
                f"visualization_total_sec={elapsed:.6f}",
            )) + "\n",
            encoding="utf-8",
        )
        print(f"Visualization time: {elapsed:.3f} s")
        print(f"Timing log: {log_path}")


if __name__ == "__main__":
    main()
