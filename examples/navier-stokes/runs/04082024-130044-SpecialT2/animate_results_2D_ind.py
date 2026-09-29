#!/usr/bin/env python3
"""Animate saved Navier–Stokes reference/prediction fields; no model required.

Run from any directory (requires numpy, matplotlib, and Pillow):
    python examples/navier-stokes/animate_results_2D_ind.py
    python examples/navier-stokes/animate_results_2D_ind.py --output flow.gif
    python examples/navier-stokes/animate_results_2D_ind.py --show

The default MP4 export requires ffmpeg; GIF export uses Pillow. The archive has
ten vorticity snapshots, not velocity vectors. Intermediate frames linearly
interpolate those snapshots for presentation; they are not new predictions.
Despite its filename, this archive's reference matches adaptation environment
3, trajectory 0. Consequently the figure makes no in-domain claim.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np


RUN_DIR = Path(__file__).resolve().parent / "runs" / "04082024-130044-SpecialT2"
DEFAULT_DATA = RUN_DIR / "results_2D_ind_train.png_data.npz"
BACKGROUND = "#0b1220"
FOREGROUND = "#e6edf5"
MUTED = "#98a9be"
ACCENT = "#57d9c5"


def load_fields(path):
    """Load the saved row-major fields and their accompanying time axis."""
    with np.load(path, allow_pickle=False) as archive:
        reference = np.asarray(archive["X"], dtype=float)
        prediction = np.asarray(archive["X_hat"], dtype=float)
        times = np.asarray(archive["t"], dtype=float) if "t" in archive else None
    if reference.shape != prediction.shape or reference.ndim not in (2, 3):
        raise ValueError("X and X_hat must have matching (time, pixels) or (time, y, x) shapes.")
    if reference.ndim == 2:
        side = int(np.sqrt(reference.shape[1]))
        if side * side != reference.shape[1]:
            raise ValueError("Flattened fields must contain a square spatial grid.")
        reference = reference.reshape(-1, side, side)
        prediction = prediction.reshape(-1, side, side)
    if len(reference) < 2 or min(reference.shape[1:]) < 2:
        raise ValueError("At least two snapshots and a 2 by 2 spatial grid are required.")
    if not (np.isfinite(reference).all() and np.isfinite(prediction).all()):
        raise ValueError("Saved fields contain NaN or infinity.")
    # The original visualizer omitted t; this run uses the same t in all splits.
    if times is None and path == DEFAULT_DATA:
        with np.load(RUN_DIR / "train_data.npz", allow_pickle=False) as archive:
            times = np.asarray(archive["t"], dtype=float)
    time_label = "Simulation time" if times is not None else "Snapshot index"
    if times is None:
        times = np.arange(len(reference), dtype=float)
    if (times.shape != (len(reference),) or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError("Times must be finite, strictly increasing, and match the snapshots.")
    return reference, prediction, times, time_label


def relative_error(reference, prediction):
    """Relative spatial L2 error; protect the zero-reference case."""
    axes = (-2, -1)
    numerator = np.sqrt(np.sum((prediction - reference) ** 2, axis=axes))
    denominator = np.sqrt(np.sum(reference ** 2, axis=axes))
    return numerator / np.maximum(denominator, np.finfo(float).eps)


def build_animation(reference, prediction, times, time_label, fps=30, duration=10):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    style = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": FOREGROUND, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#29374b", "axes.facecolor": BACKGROUND,
        "figure.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(14, 7.8), layout=None)
        grid = fig.add_gridspec(
            3, 3, left=0.065, right=0.96, bottom=0.14, top=0.80,
            height_ratios=[1, 0.045, 0.26], hspace=0.40, wspace=0.13,
        )
        axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        field_bar = fig.add_subplot(grid[1, :2])
        error_bar = fig.add_subplot(grid[1, 2])
        timeline = fig.add_subplot(grid[2, :])

        fig.text(0.065, 0.935, "NAVIER–STOKES", color=ACCENT,
                 fontsize=11, weight="bold", ha="left")
        fig.text(0.065, 0.88, "A flow, reconstructed", fontsize=27, weight="bold")
        clock = fig.text(0.96, 0.90, "", ha="right", fontsize=16,
                         fontfamily="DejaVu Sans Mono")
        fig.text(0.065, 0.035,
                 f"{reference.shape[2]} × {reference.shape[1]} vorticity fields  ·  "
                 f"{len(times)} saved snapshots  ·  linear interpolation between snapshots",
                 color=MUTED, fontsize=9)

        flow_cmap = LinearSegmentedColormap.from_list(
            "vorticity", ["#277da8", "#73c7d5", "#edf1ee", "#f4a078", "#be3c50"]
        )
        limit = max(np.max(np.abs(reference)), np.max(np.abs(prediction)), 1e-12)
        error_limit = max(np.max(np.abs(prediction - reference)), 1e-12)
        field_norm = Normalize(-limit, limit)
        error_norm = Normalize(0, error_limit)
        images = []
        for ax, field, title, cmap, norm in zip(
            axes, [reference[0], prediction[0], np.abs(prediction[0] - reference[0])],
            ["01   Reference", "02   NCF prediction", "03   Absolute error"],
            [flow_cmap, flow_cmap, "magma"], [field_norm, field_norm, error_norm],
        ):
            images.append(ax.imshow(field, origin="lower", cmap=cmap, norm=norm,
                                    interpolation="bilinear", extent=(0, 1, 0, 1)))
            ax.set_title(title, loc="left", pad=12, color=FOREGROUND, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

        for image, cax, label in [
            (images[0], field_bar, "Vorticity ω  ·  shared scale"),
            (images[2], error_bar, "|ω prediction − ω reference|"),
        ]:
            bar = fig.colorbar(image, cax=cax, orientation="horizontal")
            bar.outline.set_visible(False)
            bar.set_label(label, fontsize=9, labelpad=4)
            bar.ax.tick_params(labelsize=8, length=2)
            bar.ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))

        frame_times = np.linspace(times[0], times[-1], max(2, round(fps * duration)))
        left = np.clip(np.searchsorted(times, frame_times, side="right") - 1,
                       0, len(times) - 2)
        weights = (frame_times - times[left]) / (times[left + 1] - times[left])

        def fields_at(frame):
            i, weight = left[frame], weights[frame]
            return ((1 - weight) * reference[i] + weight * reference[i + 1],
                    (1 - weight) * prediction[i] + weight * prediction[i + 1])

        errors = np.array([100 * relative_error(*fields_at(i)) for i in range(len(frame_times))])
        timeline.fill_between(frame_times, errors, color=ACCENT, alpha=0.08)
        timeline.plot(frame_times, errors, color=ACCENT, lw=1.6, alpha=0.7)
        snapshot_errors = 100 * relative_error(reference, prediction)
        timeline.scatter(times, snapshot_errors,
                         s=16, color=ACCENT, zorder=3)
        cursor = timeline.axvline(times[0], color=FOREGROUND, lw=1, alpha=0.65)
        dot, = timeline.plot([], [], "o", color=FOREGROUND, ms=6, zorder=5)
        metric = timeline.text(1, 1.10, "", transform=timeline.transAxes,
                               ha="right", color=ACCENT, fontsize=10)
        timeline.set(xlim=(times[0], times[-1]), ylim=(0, max(max(errors.max(), snapshot_errors.max()) * 1.25, 0.1)),
                     xlabel=time_label, ylabel="Relative L2 (%)")
        timeline.spines[["top", "right"]].set_visible(False)
        timeline.grid(axis="y", color="#29374b", alpha=0.5, linewidth=0.6)
        timeline.tick_params(labelsize=9)
        timeline.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))

        def update(frame):
            ref, pred = fields_at(frame)
            for image, field in zip(images, (ref, pred, np.abs(pred - ref))):
                image.set_data(field)
            t = frame_times[frame]
            clock.set_text(f"{'t' if time_label == 'Simulation time' else 'frame'} = {t:05.2f}")
            cursor.set_xdata([t, t])
            dot.set_data([t], [errors[frame]])
            metric.set_text(f"Relative L2  {errors[frame]:.2f}%")
            return (*images, clock, cursor, dot, metric)

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frame_times),
                                  init_func=lambda: update(0), interval=1000 / fps,
                                  blit=False, repeat=True, repeat_delay=1000,
                                  cache_frame_data=False)
    return fig, animation


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="NPZ containing X and X_hat")
    parser.add_argument("--output", type=Path, help="Output .gif or .mp4 (default: MP4 beside data)")
    parser.add_argument("--fps", type=int, default=30, help="Playback frames per second (default: 30)")
    parser.add_argument("--duration", type=float, default=10, help="Playback duration in seconds (default: 10)")
    parser.add_argument("--dpi", type=int, default=120, help="Export resolution (default: 120)")
    parser.add_argument("--show", action="store_true", help="Open an interactive preview; export only if --output is set")
    args = parser.parse_args()
    if args.fps <= 0 or args.dpi <= 0 or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("fps, dpi, and duration must be positive and finite")
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter, writers

    output = args.output
    if output is None and not args.show:
        output = args.data.resolve().parent / "results_2D_ind_comparison.mp4"
    if output is not None:
        suffix = output.suffix.lower()
        if suffix not in (".gif", ".mp4"):
            parser.error("output must end in .gif or .mp4")
        if suffix == ".mp4" and not writers.is_available("ffmpeg"):
            parser.error("MP4 requires ffmpeg; install it or choose a .gif output")
    try:
        fields = load_fields(args.data.expanduser().resolve())
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    fig, animation = build_animation(*fields, fps=args.fps, duration=args.duration)
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = (PillowWriter(fps=args.fps) if output.suffix.lower() == ".gif" else
                  FFMpegWriter(fps=args.fps, codec="libx264",
                               extra_args=["-crf", "18", "-pix_fmt", "yuv420p"]))
        print(f"Rendering {args.duration:g}s at {args.fps} fps → {output}", flush=True)
        animation.save(output, writer=writer, dpi=args.dpi)
        print(f"Saved {output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
