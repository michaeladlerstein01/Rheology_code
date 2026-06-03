from pathlib import Path

import ezdxf
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from skimage import measure

try:
    from helper import build_fourier_bases, fourier_plate, init_fourier_params, params_list
except ModuleNotFoundError:
    from linear_optimization.helper import build_fourier_bases, fourier_plate, init_fourier_params, params_list

from ezdxf import units as dxf_units


# ---------------------------------------------------------------------
# Easy-to-change configuration
# ---------------------------------------------------------------------
# Plate dimensions (rows, cols)
PLATE1_H = 30
PLATE1_W = 120
PLATE2_H = 30
PLATE2_W = 230  # must be >= PLATE1_W

# Material pair friction coefficients
MU11 = 0.70  # material 1 on material 1
MU12 = 0.40  # material 1 on material 2 (and 2 on 1)
MU22 = 0.20  # material 2 on material 2

# Fourier plate generator settings
MX = 4
MY = 4
BETA = 8.0

# Training settings
EPOCHS = 3500
LEARNING_RATE = 0.2

# Visualization and GIF settings
FRAME_EVERY = 50
SHOW_EVERY = 25
GIF_DURATION_S = 0.12
GIF_PATH = Path("linear_optimization_progress.gif")
FRAME_DIR = Path("linear_results") / "frames_main_linear_friction"
SHOW_LIVE_FIGURES = True

# Optional spatial scale for x-axis labeling (set to 1.0 if each column is one unit)
COL_PITCH_MM = 1.0
ROW_PITCH_MM = 1.0

# Target curve from user points via piecewise linear stitching.
# Format: (x_norm, y_norm) where both are in [0, 1].
# - x_norm is normalized sliding position (0=start, 1=end)
# - y_norm is normalized friction level (0=mu_min, 1=mu_max)
TARGET_CLIP_TO_BOUNDS = True
TARGET_POINTS_NORM = [
    (0.00, 0.8),
    (0.10, 0.8),
    (0.20, 0.6),
    (0.30, 0.5),
    (0.40, 0.5),
    (0.50, 0.5),
    (0.60, 0.6),
    (0.70, 0.8),
    (0.80, 0.8),
    (0.90, 0.8),
    (1.00, 0.8),
]

# DXF export settings (mm)
DXF_EXPORT_DIR = Path("linear_results") / "dxf_mm_plates"
DXF_PLATE1_NAME = "linear_plate_A_mm.dxf"
DXF_PLATE2_NAME = "linear_plate_B_mm.dxf"
# Optional Fusion-Upload-compatible files (Upload can treat DXF units as cm in some flows).
DXF_EXPORT_FUSION_UPLOAD_COMPAT = True
DXF_FUSION_UPLOAD_DIR = Path("linear_results") / "dxf_fusion_upload_compat"
DXF_PLATE1_UPLOAD_NAME = "linear_plate_A_upload_compat.dxf"
DXF_PLATE2_UPLOAD_NAME = "linear_plate_B_upload_compat.dxf"
MATERIAL1_THRESHOLD = 0.5
MATERIAL1_NAME = "Rough surface"
MATERIAL2_NAME = "Smooth surface"
MATERIAL1_COLOR = "#E67E22"  # orange
MATERIAL2_COLOR = "#2E86C1"  # blue

# Keep at 1.0 for true mm coordinates.
DXF_COORDINATE_SCALE = 1.0
# For Fusion Upload path that may assume cm, 0.1 gives correct physical mm size.
DXF_FUSION_UPLOAD_SCALE = 0.1


def friction_curve_sum(
    moving_plate: torch.Tensor,
    fixed_plate: torch.Tensor,
    shifts: torch.Tensor,
    mu11: float,
    mu12: float,
    mu22: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute friction curve as requested:
      at each shift, compute local coefficient map and sum the average coefficient.

    Plate values in [0,1] are interpreted as fraction of material 1.
    """
    h1, w1 = moving_plate.shape
    h2, w2 = fixed_plate.shape
    h = min(h1, h2)

    mu_sum_vals = []
    mu_avg_vals = []
    for s in shifts.tolist():
        s = int(s)
        sl = fixed_plate[:h, s : s + w1]
        mv = moving_plate[:h, :w1]

        # Material 1 fraction
        p1 = mv.clamp(0.0, 1.0)
        p2 = sl.clamp(0.0, 1.0)

        # Expected local friction coefficient from material fractions
        mu_local = (
            mu11 * (p1 * p2)
            + mu12 * (p1 * (1.0 - p2) + (1.0 - p1) * p2)
            + mu22 * ((1.0 - p1) * (1.0 - p2))
        )

        mu_avg = mu_local.mean()
        # "sum the average coefficient" at this step over the overlap region
        mu_sum = mu_avg * mu_local.numel()

        mu_avg_vals.append(mu_avg)
        mu_sum_vals.append(mu_sum)

    return torch.stack(mu_sum_vals), torch.stack(mu_avg_vals)


def build_target_curve(shifts: torch.Tensor, mu_min: float, mu_max: float, contact_cells: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build target from user-provided points using piecewise linear interpolation.
    """
    pts = np.asarray(TARGET_POINTS_NORM, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("TARGET_POINTS_NORM must be a list of (x_norm, y_norm) pairs.")
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points in TARGET_POINTS_NORM.")

    x_pts = pts[:, 0]
    y_pts = pts[:, 1]

    order = np.argsort(x_pts)
    x_pts = x_pts[order]
    y_pts = y_pts[order]

    if np.any(np.diff(x_pts) <= 0.0):
        raise ValueError("TARGET_POINTS_NORM x-values must be strictly increasing.")
    if x_pts[0] < 0.0 or x_pts[-1] > 1.0:
        raise ValueError("TARGET_POINTS_NORM x-values must be in [0, 1].")

    # Normalized evaluation axis over the active shift range
    if shifts.numel() <= 1:
        x_eval = np.zeros((int(shifts.numel()),), dtype=np.float64)
    else:
        s_np = shifts.detach().cpu().numpy().astype(np.float64)
        x_eval = (s_np - s_np.min()) / (s_np.max() - s_np.min())

    # Piecewise linear target stitched from user points.
    y_eval = np.interp(
        x_eval,
        x_pts,
        y_pts,
        left=y_pts[0],
        right=y_pts[-1],
    )

    if TARGET_CLIP_TO_BOUNDS:
        y_eval = np.clip(y_eval, 0.0, 1.0)

    span = mu_max - mu_min
    target_mu_avg_np = mu_min + span * y_eval
    target_mu_avg = torch.as_tensor(target_mu_avg_np, dtype=torch.float32, device=shifts.device)
    target_mu_sum = target_mu_avg * float(contact_cells)
    return target_mu_sum, target_mu_avg


def _closed_contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded.astype(np.float32), 0.5)
    closed = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        cnt = cnt - 1.0
        if np.linalg.norm(cnt[0] - cnt[-1]) > 1e-6:
            continue
        closed.append(cnt)
    return closed


def export_linear_plate_to_dxf_mm(
    plate: torch.Tensor,
    out_path: Path,
    col_pitch_mm: float,
    row_pitch_mm: float,
    material1_threshold: float = 0.5,
    coordinate_scale: float = 1.0,
) -> None:
    """
    Export a rectangular plate in mm:
    - OUTER_RECT layer: full plate boundary
    - MAT1_REGIONS layer: contours of material-1 regions (plate >= threshold)
    """
    plate_np = plate.detach().cpu().numpy()
    mat1_mask = plate_np >= material1_threshold
    h, w = mat1_mask.shape
    width_mm = w * col_pitch_mm * coordinate_scale
    height_mm = h * row_pitch_mm * coordinate_scale

    doc = ezdxf.new(dxfversion="R2013")
    doc.units = dxf_units.MM
    doc.header["$INSUNITS"] = dxf_units.MM
    doc.header["$MEASUREMENT"] = 1  # metric
    doc.header["$LUNITS"] = 2       # decimal display
    msp = doc.modelspace()

    if "OUTER_RECT" not in doc.layers:
        doc.layers.add(name="OUTER_RECT", color=3)
    if "MAT1_REGIONS" not in doc.layers:
        doc.layers.add(name="MAT1_REGIONS", color=5)

    # Outer rectangle boundary
    outer_pts = [(0.0, 0.0), (width_mm, 0.0), (width_mm, height_mm), (0.0, height_mm)]
    msp.add_lwpolyline(outer_pts, format="xy", close=True, dxfattribs={"layer": "OUTER_RECT"})

    contours = _closed_contours_from_mask(mat1_mask)
    if not contours:
        # Fallback: if threshold makes a fully uniform mask, try median split for visible pattern export.
        alt_threshold = float(np.median(plate_np))
        if abs(alt_threshold - material1_threshold) > 1e-6:
            alt_mask = plate_np >= alt_threshold
            contours = _closed_contours_from_mask(alt_mask)
            if contours:
                print(
                    f"DXF note: no contours at threshold={material1_threshold:.3f}; "
                    f"used median threshold={alt_threshold:.3f} for pattern export."
                )

    # Material-1 stitched contours
    contour_count = 0
    for cnt in contours:
        rr = np.clip(cnt[:, 0], 0.0, float(h))
        cc = np.clip(cnt[:, 1], 0.0, float(w))
        # cnt columns -> x, rows -> y (flip y so origin is bottom-left)
        x_mm = cc * col_pitch_mm * coordinate_scale
        y_mm = (h - rr) * row_pitch_mm * coordinate_scale
        pts = list(zip(x_mm.astype(float), y_mm.astype(float)))
        if len(pts) >= 3:
            msp.add_lwpolyline(pts, format="xy", close=True, dxfattribs={"layer": "MAT1_REGIONS"})
            contour_count += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(out_path))
    print(
        f"DXF exported: {out_path.resolve()} | pattern contours: {contour_count} | "
        f"units=mm | coord_scale={coordinate_scale}"
    )


def main():
    if PLATE2_W < PLATE1_W:
        raise ValueError("PLATE2_W must be >= PLATE1_W for sliding window overlap.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    h_overlap = min(PLATE1_H, PLATE2_H)
    contact_cells = h_overlap * PLATE1_W

    # Shift axis: exact column shifts where plate1 fully overlaps a slice of plate2
    max_shift = PLATE2_W - PLATE1_W
    shifts = torch.arange(0, max_shift + 1, device=device, dtype=torch.int64)
    x_mm = shifts.to(torch.float32) * COL_PITCH_MM

    print(
        f"Friction bounds (avg mu): [{MU22:.3f}, {MU11:.3f}] | "
        f"contact cells/step: {contact_cells} | "
        f"sum bounds: [{MU22 * contact_cells:.3f}, {MU11 * contact_cells:.3f}]"
    )

    target_mu_sum, target_mu_avg = build_target_curve(
        shifts=shifts.to(torch.float32),
        mu_min=min(MU11, MU12, MU22),
        mu_max=max(MU11, MU12, MU22),
        contact_cells=contact_cells,
    )
    target_mu_sum = target_mu_sum.to(device=device, dtype=dtype)
    target_mu_avg = target_mu_avg.to(device=device, dtype=dtype)

    # Build plates from helper logic
    bases1, *_ = build_fourier_bases(PLATE1_H, PLATE1_W, MX, MY, device=device, dtype=dtype)
    bases2, *_ = build_fourier_bases(PLATE2_H, PLATE2_W, MX, MY, device=device, dtype=dtype)
    plate1_params = init_fourier_params(MX, MY, init_scale=0.10, device=device, dtype=dtype)
    plate2_params = init_fourier_params(MX, MY, init_scale=0.10, device=device, dtype=dtype)

    optimizer = torch.optim.Adam(
        params_list(plate1_params) + params_list(plate2_params),
        lr=LEARNING_RATE,
    )

    # Live plot state (persistent figure updated in-place)
    material_cmap = ListedColormap([MATERIAL2_COLOR, MATERIAL1_COLOR])
    material_legend = [
        Patch(facecolor=MATERIAL1_COLOR, edgecolor="black", label=MATERIAL1_NAME),
        Patch(facecolor=MATERIAL2_COLOR, edgecolor="black", label=MATERIAL2_NAME),
    ]

    if SHOW_LIVE_FIGURES:
        plt.ion()
        live_fig = plt.figure(figsize=(12, 7))
        try:
            live_fig.canvas.manager.set_window_title("Linear Friction Optimization (Live)")
        except Exception:
            pass
        live_ax1 = live_fig.add_subplot(2, 2, 1)
        live_ax2 = live_fig.add_subplot(2, 2, 2)
        live_ax3 = live_fig.add_subplot(2, 1, 2)
        live_im1 = None
        live_im2 = None
        live_line_target = None
        live_line_pred = None
    else:
        live_fig = None
        live_ax1 = live_ax2 = live_ax3 = None
        live_im1 = live_im2 = None
        live_line_target = live_line_pred = None

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAME_DIR.glob("frame_*.png"):
        old.unlink()
    frame_paths: list[Path] = []

    losses = []
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        plate1 = fourier_plate(plate1_params, bases1, beta=BETA, hard=False)
        plate2 = fourier_plate(plate2_params, bases2, beta=BETA, hard=False)

        pred_mu_sum, pred_mu_avg = friction_curve_sum(
            moving_plate=plate1,
            fixed_plate=plate2,
            shifts=shifts,
            mu11=MU11,
            mu12=MU12,
            mu22=MU22,
        )

        # Main fit in "sum of average coefficients" space
        loss_main = F.mse_loss(pred_mu_sum, target_mu_sum)
        # Keep modulation depth from collapsing
        loss_amp = (pred_mu_avg.std() - target_mu_avg.std()) ** 2
        # Encourage material separation (less gray)
        bin_loss = (plate1 * (1.0 - plate1)).mean() + (plate2 * (1.0 - plate2)).mean()

        loss = loss_main + 0.5 * loss_amp + 0.02 * bin_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss={loss.item():.6f} | "
                f"MSE(sum)={loss_main.item():.6f} | "
                f"Std(avg_mu)={pred_mu_avg.std().item():.6f}"
            )

        should_save = (epoch % FRAME_EVERY == 0) or (epoch == EPOCHS - 1)
        should_show = SHOW_LIVE_FIGURES and ((epoch % SHOW_EVERY == 0) or (epoch == EPOCHS - 1))

        if should_save or should_show:
            p1_np = plate1.detach().cpu().numpy()
            p2_np = plate2.detach().cpu().numpy()
            pred_np = pred_mu_sum.detach().cpu().numpy()
            tgt_np = target_mu_sum.detach().cpu().numpy()
            x_np = x_mm.detach().cpu().numpy()

            if SHOW_LIVE_FIGURES:
                if live_im1 is None:
                    p1_vis = (p1_np >= MATERIAL1_THRESHOLD).astype(np.uint8)
                    p2_vis = (p2_np >= MATERIAL1_THRESHOLD).astype(np.uint8)
                    live_im1 = live_ax1.imshow(
                        p1_vis, origin="lower", cmap=material_cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest"
                    )
                    live_ax1.set_title("Sliding Plate")
                    live_ax1.set_xticks([])
                    live_ax1.set_yticks([])
                    live_ax1.legend(handles=material_legend, loc="upper right", fontsize=8, framealpha=0.9)

                    live_im2 = live_ax2.imshow(
                        p2_vis, origin="lower", cmap=material_cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest"
                    )
                    live_ax2.set_title("Fixed Plate")
                    live_ax2.set_xticks([])
                    live_ax2.set_yticks([])
                    live_ax2.legend(handles=material_legend, loc="upper right", fontsize=8, framealpha=0.9)

                    (live_line_target,) = live_ax3.plot(x_np, tgt_np, label="Target friction sum", linewidth=2)
                    (live_line_pred,) = live_ax3.plot(x_np, pred_np, "--", label="Predicted friction sum")
                    live_ax3.set_title("Friction Response vs Sliding Position")
                    live_ax3.set_xlabel("Sliding offset (mm)")
                    live_ax3.set_ylabel("Sum of average coefficients")
                    live_ax3.grid(True)
                    live_ax3.legend()
                else:
                    live_im1.set_data((p1_np >= MATERIAL1_THRESHOLD).astype(np.uint8))
                    live_im2.set_data((p2_np >= MATERIAL1_THRESHOLD).astype(np.uint8))
                    live_line_target.set_data(x_np, tgt_np)
                    live_line_pred.set_data(x_np, pred_np)
                    live_ax3.relim()
                    live_ax3.autoscale_view(scalex=False, scaley=True)

                live_fig.tight_layout()
                live_fig.canvas.draw()
                live_fig.canvas.flush_events()

            # Separate static figure for GIF frames
            if should_save:
                p1_vis = (p1_np >= MATERIAL1_THRESHOLD).astype(np.uint8)
                p2_vis = (p2_np >= MATERIAL1_THRESHOLD).astype(np.uint8)
                frame_fig = plt.figure(figsize=(12, 7))
                ax1 = frame_fig.add_subplot(2, 2, 1)
                ax1.imshow(p1_vis, origin="lower", cmap=material_cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
                ax1.set_title("Sliding Plate")
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.legend(handles=material_legend, loc="upper right", fontsize=8, framealpha=0.9)

                ax2 = frame_fig.add_subplot(2, 2, 2)
                ax2.imshow(p2_vis, origin="lower", cmap=material_cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
                ax2.set_title("Fixed Plate")
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.legend(handles=material_legend, loc="upper right", fontsize=8, framealpha=0.9)

                ax3 = frame_fig.add_subplot(2, 1, 2)
                ax3.plot(x_np, tgt_np, label="Target friction sum", linewidth=2)
                ax3.plot(x_np, pred_np, "--", label="Predicted friction sum")
                ax3.set_title("Friction Response vs Sliding Position")
                ax3.set_xlabel("Sliding offset (mm)")
                ax3.set_ylabel("Sum of average coefficients")
                ax3.grid(True)
                ax3.legend()
                frame_fig.tight_layout()

                frame_path = FRAME_DIR / f"frame_{epoch:05d}.png"
                frame_fig.savefig(frame_path, dpi=150)
                frame_paths.append(frame_path)
                plt.close(frame_fig)

            if should_show and SHOW_LIVE_FIGURES:
                plt.pause(0.001)

    # Final loss plot
    if SHOW_LIVE_FIGURES:
        plt.figure(figsize=(7, 3))
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
        plt.close()
        if live_fig is not None:
            plt.ioff()
            # keep final live figure open at the end
            live_fig.show()

    if frame_paths:
        with imageio.get_writer(GIF_PATH, mode="I", duration=GIF_DURATION_S, loop=0) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        print(f"GIF exported: {GIF_PATH.resolve()}")

    # Export final optimized linear plates as DXF (mm) in a separate folder.
    export_linear_plate_to_dxf_mm(
        plate=plate1,
        out_path=DXF_EXPORT_DIR / DXF_PLATE1_NAME,
        col_pitch_mm=COL_PITCH_MM,
        row_pitch_mm=ROW_PITCH_MM,
        material1_threshold=MATERIAL1_THRESHOLD,
        coordinate_scale=DXF_COORDINATE_SCALE,
    )
    export_linear_plate_to_dxf_mm(
        plate=plate2,
        out_path=DXF_EXPORT_DIR / DXF_PLATE2_NAME,
        col_pitch_mm=COL_PITCH_MM,
        row_pitch_mm=ROW_PITCH_MM,
        material1_threshold=MATERIAL1_THRESHOLD,
        coordinate_scale=DXF_COORDINATE_SCALE,
    )

    if DXF_EXPORT_FUSION_UPLOAD_COMPAT:
        export_linear_plate_to_dxf_mm(
            plate=plate1,
            out_path=DXF_FUSION_UPLOAD_DIR / DXF_PLATE1_UPLOAD_NAME,
            col_pitch_mm=COL_PITCH_MM,
            row_pitch_mm=ROW_PITCH_MM,
            material1_threshold=MATERIAL1_THRESHOLD,
            coordinate_scale=DXF_FUSION_UPLOAD_SCALE,
        )
        export_linear_plate_to_dxf_mm(
            plate=plate2,
            out_path=DXF_FUSION_UPLOAD_DIR / DXF_PLATE2_UPLOAD_NAME,
            col_pitch_mm=COL_PITCH_MM,
            row_pitch_mm=ROW_PITCH_MM,
            material1_threshold=MATERIAL1_THRESHOLD,
            coordinate_scale=DXF_FUSION_UPLOAD_SCALE,
        )


if __name__ == "__main__":
    main()
