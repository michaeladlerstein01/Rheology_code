import numpy as np
import os
import torch
from compute_wimg import *
import imageio.v2 as imageio
from ezdxf import units as dxf_units
from matplotlib import pyplot as plt
from pathlib import Path


# Piecewise-linear target curve points.
# x is normalized angle (0.25 = 90 degrees), y is normalized torque level.
TARGET_POINTS_NORM = [
    (0.00, 0.3),
    (0.10, 0.3),
    (0.22, 0.6),
    (0.35, 0.5),
    (0.50, 0.5),
    (0.64, 0.3),
    (0.78, 0.3),
    (0.90, 0.4),
    (1.00, 0.4),
]
TARGET_MARGIN_FRACTION = 0.08
NUM_EPOCHS = int(os.getenv("TEST3_NUM_EPOCHS", "5000"))
LEARNING_RATE = float(os.getenv("TEST3_LEARNING_RATE", "0.012"))
EXPLORATION_EPOCHS = int(os.getenv("TEST3_EXPLORATION_EPOCHS", "5000"))
EXPLORATION_NOISE_STD = float(os.getenv("TEST3_EXPLORATION_NOISE_STD", "0.015"))
EXPLORATION_NOISE_EVERY = int(os.getenv("TEST3_EXPLORATION_NOISE_EVERY", "25"))
GIF_EVERY = int(os.getenv("TEST3_GIF_EVERY", "100"))
DISPLAY_EVERY = int(os.getenv("TEST3_DISPLAY_EVERY", "50"))
GIF_FRAME_DURATION_S = float(os.getenv("TEST3_GIF_FRAME_DURATION_S", "0.16"))
SHOW_LIVE_OPTIMIZATION = os.getenv("TEST3_SHOW_LIVE", "1").lower() not in {"0", "false", "no"}
PLATE_DIAMETER_MM = 50.0
PLATE_RADIUS_MM = PLATE_DIAMETER_MM / 2.0
OUTER_RADIUS_M = PLATE_RADIUS_MM / 1000.0


def radial_edge_unimodality_loss(
    img: torch.Tensor,
    radius_px: int,
    num_angles: int = 72,
    num_r: int | None = None,
) -> torch.Tensor:
    """
    Encourage one dominant edge crossing per ray from the disk center.
    """
    if img.ndim == 2:
        x = img.unsqueeze(0).unsqueeze(0)
    else:
        x = img

    if num_r is None:
        num_r = radius_px + 1

    theta = torch.linspace(0.0, 2 * math.pi, steps=num_angles + 1, device=x.device)[:-1]
    r = torch.linspace(0.0, float(radius_px), steps=num_r, device=x.device)

    rr = r.view(1, 1, -1).expand(1, theta.numel(), -1)
    tt = theta.view(1, -1, 1).expand(1, theta.numel(), r.numel())

    gx = (rr * torch.cos(tt)) / radius_px
    gy = (rr * torch.sin(tt)) / radius_px
    grid = torch.stack([gx, gy], dim=-1)

    pr = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    pr = pr[0, 0]

    dp = pr[:, 1:] - pr[:, :-1]
    w = 4.0 * pr * (1.0 - pr)
    w_mid = 0.5 * (w[:, 1:] + w[:, :-1])
    edge = dp.abs() * w_mid

    eps = 1e-8
    p_edge = edge / (edge.sum(dim=1, keepdim=True) + eps)
    entropy = -(p_edge * (p_edge + eps).log()).sum(dim=1)
    entropy = entropy / math.log(max(2, edge.shape[1]))
    return entropy.mean()


def build_disk_geometry(grid_size: int, outer_radius_m: float, device: torch.device):
    radius_px = grid_size // 2 - 4
    coord = torch.linspace(-radius_px, radius_px, grid_size, device=device)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    r_px = torch.sqrt(xx**2 + yy**2)
    disk_mask = (r_px <= radius_px).float()

    px_to_m = outer_radius_m / float(radius_px)
    r_m = r_px * px_to_m
    dA_m2 = px_to_m**2
    return disk_mask, r_m, dA_m2


def disk_friction_torque_at_angle(
    pattern_A: torch.Tensor,
    pattern_B: torch.Tensor,
    angle_deg: float,
    disk_mask: torch.Tensor,
    radius_map_m: torch.Tensor,
    pixel_area_m2: float,
    pressure_pa: float,
    mu_tpu_tpu: float,
    mu_pla_pla: float,
    mu_tpu_pla: float,
) -> torch.Tensor:
    """
    Disk friction model:
      dT = mu_local * p * r * dA
      T(theta) = integral_A dT

    Material map convention:
      black -> TPU
      white -> PLA
    with plate values in [0,1] interpreted as PLA fraction.
    """
    if pattern_A.ndim == 2:
        plate_A = pattern_A.unsqueeze(0).unsqueeze(0)
    else:
        plate_A = pattern_A
    if pattern_B.ndim == 2:
        plate_B = pattern_B.unsqueeze(0).unsqueeze(0)
    else:
        plate_B = pattern_B

    plate_B_rot = rotate_tensor(plate_B, angle_deg)[0, 0]
    plate_A = plate_A[0, 0]

    # White = PLA fraction, Black = TPU fraction
    pla_A = (plate_A * disk_mask).clamp(0.0, 1.0)
    pla_B = (plate_B_rot * disk_mask).clamp(0.0, 1.0)
    tpu_A = (disk_mask - pla_A).clamp(0.0, 1.0)
    tpu_B = (disk_mask - pla_B).clamp(0.0, 1.0)
    mu_local = (
        mu_pla_pla * (pla_A * pla_B)
        + mu_tpu_tpu * (tpu_A * tpu_B)
        + mu_tpu_pla * (pla_A * tpu_B + tpu_A * pla_B)
    )

    torque_density = mu_local * pressure_pa * radius_map_m * pixel_area_m2
    return torque_density.sum()


def disk_friction_sweep_torch(
    pattern_A: torch.Tensor,
    pattern_B: torch.Tensor,
    angles_deg: torch.Tensor,
    disk_mask: torch.Tensor,
    radius_map_m: torch.Tensor,
    pixel_area_m2: float,
    pressure_pa: float,
    mu_tpu_tpu: float,
    mu_pla_pla: float,
    mu_tpu_pla: float,
) -> torch.Tensor:
    return torch.stack(
        [
            disk_friction_torque_at_angle(
                pattern_A=pattern_A,
                pattern_B=pattern_B,
                angle_deg=a.item(),
                disk_mask=disk_mask,
                radius_map_m=radius_map_m,
                pixel_area_m2=pixel_area_m2,
                pressure_pa=pressure_pa,
                mu_tpu_tpu=mu_tpu_tpu,
                mu_pla_pla=mu_pla_pla,
                mu_tpu_pla=mu_tpu_pla,
            )
            for a in angles_deg
        ]
    )


def build_scaled_sinusoidal_target(
    x: torch.Tensor,
    torque_min: float,
    torque_max: float,
) -> torch.Tensor:
    """
    Build a smooth custom sinusoidal target inside the achievable torque range.
    """
    span = torque_max - torque_min
    center = 0.5 * (torque_max + torque_min)

    target = (
        center
        + 0.24 * span * torch.sin(1.0 * x + 0.35)
        + 0.08 * span * torch.sin(2.0 * x - 1.10)
    )

    safety_margin = 0.08 * span
    return target.clamp(torque_min + safety_margin, torque_max - safety_margin)


def build_piecewise_linear_target(
    angles_deg: torch.Tensor,
    torque_min: float,
    torque_max: float,
    target_points_norm: list[tuple[float, float]] = TARGET_POINTS_NORM,
    margin_fraction: float = TARGET_MARGIN_FRACTION,
) -> torch.Tensor:
    """
    Build a target curve by stitching straight lines between user-defined points.

    Points are normalized:
      x in [0, 1] maps to angle from 0 to 360 degrees.
      y in [0, 1] maps to the usable torque range.
    """
    pts = np.asarray(target_points_norm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError("target_points_norm must contain at least two (x_norm, y_norm) pairs.")

    order = np.argsort(pts[:, 0])
    x_pts = pts[order, 0]
    y_pts = pts[order, 1]

    if np.any(np.diff(x_pts) <= 0.0):
        raise ValueError("target point x-values must be strictly increasing.")
    if x_pts[0] < 0.0 or x_pts[-1] > 1.0:
        raise ValueError("target point x-values must stay inside [0, 1].")

    y_pts = np.clip(y_pts, 0.0, 1.0)
    x_eval = angles_deg.detach().cpu().numpy().astype(np.float64) / 360.0
    y_eval = np.interp(x_eval, x_pts, y_pts, left=y_pts[0], right=y_pts[-1])

    span = torque_max - torque_min
    usable_min = torque_min + margin_fraction * span
    usable_max = torque_max - margin_fraction * span
    target_np = usable_min + (usable_max - usable_min) * y_eval
    return torch.as_tensor(target_np, dtype=torch.float32, device=angles_deg.device)


def binary_push_loss(pattern: torch.Tensor, disk_mask: torch.Tensor) -> torch.Tensor:
    """
    Encourage black/white (TPU/PLA) plate patterns instead of gray mixtures.
    """
    p = (pattern * disk_mask).clamp(0.0, 1.0)
    return (p * (1.0 - p)).mean()


def _closed_contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    """
    Return only closed contours.
    We pad the mask first so contours near the edge can still close correctly.
    """
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded.astype(np.float32), 0.5)

    closed = []
    H, W = mask.shape
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        cnt = cnt - 1.0  # remove padding shift

        # Open contours from marching squares have distinct endpoints.
        if np.linalg.norm(cnt[0] - cnt[-1]) > 1e-6:
            continue

        # Reject anything touching the image border (likely export artifact).
        rr = cnt[:, 0]
        cc = cnt[:, 1]
        if rr.min() <= -1e-6 or cc.min() <= -1e-6 or rr.max() >= (H - 1) + 1e-6 or cc.max() >= (W - 1) + 1e-6:
            continue

        closed.append(cnt)

    return closed


def export_plate_pattern_to_dxf(
    pattern: torch.Tensor,
    disk_mask: torch.Tensor,
    out_dxf_path: str,
    plate_radius_mm: float = PLATE_RADIUS_MM,
    pla_threshold: float = 0.5,
):
    """
    Export one optimized plate pattern to DXF for Fusion.
    - `DISK_OUTER` layer: full circular plate boundary
    - `PLA_REGIONS` layer: boundaries of white regions (PLA)
    """
    pattern_np = pattern.detach().cpu().numpy()
    disk_np = (disk_mask.detach().cpu().numpy() > 0.5)
    pla_mask = (pattern_np >= pla_threshold) & disk_np

    H, W = pla_mask.shape
    center_r = (H - 1) / 2.0
    center_c = (W - 1) / 2.0
    radius_px = H // 2 - 4
    mm_per_px = plate_radius_mm / float(radius_px)

    doc = ezdxf.new(dxfversion="R2013")
    doc.units = dxf_units.MM
    doc.header["$INSUNITS"] = dxf_units.MM
    doc.header["$MEASUREMENT"] = 1
    doc.header["$LUNITS"] = 2
    msp = doc.modelspace()

    if "DISK_OUTER" not in doc.layers:
        doc.layers.add(name="DISK_OUTER", color=3)
    if "PLA_REGIONS" not in doc.layers:
        doc.layers.add(name="PLA_REGIONS", color=5)

    msp.add_circle((0.0, 0.0), plate_radius_mm, dxfattribs={"layer": "DISK_OUTER"})

    for cnt in _closed_contours_from_mask(pla_mask):
        x_mm = (cnt[:, 1] - center_c) * mm_per_px
        y_mm = -(cnt[:, 0] - center_r) * mm_per_px
        pts = []
        for x_i, y_i in zip(x_mm, y_mm):
            r_i = math.hypot(float(x_i), float(y_i))
            # Keep contours strictly inside the plate radius.
            if r_i > plate_radius_mm and r_i > 1e-9:
                s = plate_radius_mm / r_i
                x_i *= s
                y_i *= s
            pts.append((float(x_i), float(y_i)))

        # Guard against accidental long, out-of-disk artifacts.
        r_max = max(math.hypot(p[0], p[1]) for p in pts)
        if r_max > plate_radius_mm + 1.5 * mm_per_px:
            continue

        if len(pts) >= 3:
            msp.add_lwpolyline(pts, format="xy", close=True, dxfattribs={"layer": "PLA_REGIONS"})

    doc.saveas(out_dxf_path)
    print(f"DXF exported: {out_dxf_path} | diameter={2.0 * plate_radius_mm:.1f} mm")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Template-style optimization constants
    M2, N2, grid_size = 8, 4, 64
    beta = 8.0

    # Disk friction model constants
    outer_radius_m = OUTER_RADIUS_M  # 50 mm diameter plate
    pressure_pa = 20000.0  # uniform contact pressure
    mu_tpu_tpu = 0.60
    mu_pla_pla = 0.30
    # Keep cross-material friction away from the arithmetic mean.
    # If mu_tpu_pla == 0.5*(mu_tpu_tpu + mu_pla_pla), angle interaction cancels.
    # Lower values here increase modulation depth in angle response.
    mu_tpu_pla = 0.30

    interaction_gain = mu_tpu_tpu - 2.0 * mu_tpu_pla + mu_pla_pla
    if abs(interaction_gain) < 1e-6:
        raise ValueError(
            "Degenerate friction coefficients: choose mu_tpu_pla different "
            "from 0.5*(mu_tpu_tpu + mu_pla_pla) to keep angle dependence."
        )

    disk_mask, radius_map_m, pixel_area_m2 = build_disk_geometry(
        grid_size=grid_size,
        outer_radius_m=outer_radius_m,
        device=device,
    )

    # Use one consistent angle grid (no duplicated 0/360 endpoint).
    num_angles = 72
    angles_deg = torch.linspace(0.0, 360.0, steps=num_angles + 1, device=device)[:-1]

    # Custom piecewise-linear target friction/torque curve scaled to achievable bounds.
    base_torque_scale = pressure_pa * pixel_area_m2 * torch.sum(radius_map_m * disk_mask).item()
    torque_min = mu_pla_pla * base_torque_scale
    torque_max = mu_tpu_tpu * base_torque_scale
    torque_span = torque_max - torque_min
    target_torque = build_piecewise_linear_target(
        angles_deg=angles_deg,
        torque_min=torque_min,
        torque_max=torque_max,
    )
    target_torque_norm = (target_torque - torque_min) / torque_span

    print(
        f"Piecewise-linear target torque range: "
        f"[{target_torque.min().item():.4f}, {target_torque.max().item():.4f}] N.m "
        f"(model bounds: [{torque_min:.4f}, {torque_max:.4f}] N.m), "
        f"interaction_gain={interaction_gain:.4f}"
    )
    print(
        f"Optimization settings: epochs={NUM_EPOCHS}, lr={LEARNING_RATE}, "
        f"exploration_epochs={EXPLORATION_EPOCHS}, "
        f"noise_std={EXPLORATION_NOISE_STD}, noise_every={EXPLORATION_NOISE_EVERY}"
    )

    # Trainable Fourier coefficients for both disks
    c_cos_A = torch.nn.Parameter(torch.randn(N2 + 1, M2 + 1, device=device))
    c_sin_A = torch.nn.Parameter(torch.randn(N2 + 1, M2 + 1, device=device))
    c_cos_B = torch.nn.Parameter(torch.randn(N2 + 1, M2 + 1, device=device))
    c_sin_B = torch.nn.Parameter(torch.randn(N2 + 1, M2 + 1, device=device))
    with torch.no_grad():
        c_sin_A[:, 0] = 0.0
        c_sin_B[:, 0] = 0.0

    optim_params = [c_cos_A, c_sin_A, c_cos_B, c_sin_B]
    optimizer = torch.optim.Adam(optim_params, lr=LEARNING_RATE)

    loss_history = []
    rad_weight_max = 0.02
    rad_warmup_epochs = 4000
    amp_loss_weight = 2.0
    deriv_loss_weight = 0.6  #0.8
    peak_mse_gain = 3 #2.5 
    binary_loss_weight = 0.01
    num_epochs = NUM_EPOCHS
    gif_every = GIF_EVERY
    display_every = DISPLAY_EVERY
    gif_frame_duration_s = GIF_FRAME_DURATION_S
    show_live_optimization = SHOW_LIVE_OPTIMIZATION

    frame_dir = Path("results") / "test3_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.png"):
        old_frame.unlink()
    frame_paths: list[Path] = []

    angle_deg_np = angles_deg.detach().cpu().numpy()
    target_torque_np = target_torque.detach().cpu().numpy()

    if show_live_optimization:
        plt.ion()
        live_fig = plt.figure(figsize=(12, 6))
        try:
            live_fig.canvas.manager.set_window_title("Test3 Rotational Optimization")
        except Exception:
            pass

        live_ax1 = live_fig.add_subplot(2, 3, 1)
        live_im_A = live_ax1.imshow(np.zeros((grid_size, grid_size)), cmap="gray", origin="lower", vmin=0, vmax=1)
        live_ax1.set_title("Disk A (white=PLA, black=TPU)")
        live_ax1.axis("off")

        live_ax2 = live_fig.add_subplot(2, 3, 2)
        live_im_B = live_ax2.imshow(np.zeros((grid_size, grid_size)), cmap="gray", origin="lower", vmin=0, vmax=1)
        live_ax2.set_title("Disk B")
        live_ax2.axis("off")

        live_ax3 = live_fig.add_subplot(2, 1, 2)
        live_ax3.plot(angle_deg_np, target_torque_np, label="Target Piecewise-Linear Torque", linewidth=2)
        (live_pred_line,) = live_ax3.plot(angle_deg_np, np.zeros_like(angle_deg_np), "--", label="Predicted Torque")
        live_ax3.set_title("Disk Friction Torque vs Rotation Angle")
        live_ax3.set_xlabel("Angle (deg)")
        live_ax3.set_ylabel("Torque (N.m, scaled)")
        live_ax3.set_xlim(0.0, 360.0)
        live_ax3.set_xticks(np.arange(0.0, 361.0, 45.0))
        live_ax3.grid(True)
        live_ax3.legend()
        live_fig.tight_layout()
        live_fig.show()
    else:
        live_fig = None
        live_im_A = live_im_B = live_pred_line = None
        live_ax2 = live_ax3 = None

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        pattern_A = generate_pattern_torch(grid_size, M2, N2, beta, c_cos_A, c_sin_A)
        pattern_B = generate_pattern_torch(grid_size, M2, N2, beta, c_cos_B, c_sin_B)

        torque_pred = disk_friction_sweep_torch(
            pattern_A=pattern_A,
            pattern_B=pattern_B,
            angles_deg=angles_deg,
            disk_mask=disk_mask,
            radius_map_m=radius_map_m,
            pixel_area_m2=pixel_area_m2,
            pressure_pa=pressure_pa,
            mu_tpu_tpu=mu_tpu_tpu,
            mu_pla_pla=mu_pla_pla,
            mu_tpu_pla=mu_tpu_pla,
        )
        torque_pred_norm = (torque_pred - torque_min) / torque_span

        # Fit in normalized torque space so 50 mm plates optimize with healthy gradients.
        dev = (target_torque_norm - target_torque_norm.mean()).abs()
        dev = dev / (dev.max() + 1e-8)
        peak_weights = 1.0 + peak_mse_gain * dev
        loss_weighted_mse = ((torque_pred_norm - target_torque_norm) ** 2 * peak_weights).mean()

        # Match global modulation amplitude and local slope/shape.
        loss_amp = (torque_pred_norm.std() - target_torque_norm.std()) ** 2
        loss_deriv = F.mse_loss(torch.diff(torque_pred_norm), torch.diff(target_torque_norm))

        radius_px = grid_size // 2 - 4
        rad_loss_A = radial_edge_unimodality_loss(pattern_A, radius_px=radius_px, num_angles=72)
        rad_loss_B = radial_edge_unimodality_loss(pattern_B, radius_px=radius_px, num_angles=72)
        rad_weight = rad_weight_max * min(1.0, epoch / float(rad_warmup_epochs))
        rad_loss = rad_weight * (rad_loss_A + rad_loss_B)

        # Push solutions toward discrete TPU/PLA regions to increase friction contrast.
        bin_loss = binary_push_loss(pattern_A, disk_mask) + binary_push_loss(pattern_B, disk_mask)

        loss_fit = loss_weighted_mse + amp_loss_weight * loss_amp + deriv_loss_weight * loss_deriv
        loss = loss_fit + rad_loss + binary_loss_weight * bin_loss
        loss.backward()

        with torch.no_grad():
            c_sin_A.grad[:, 0] = 0.0
            c_sin_B.grad[:, 0] = 0.0

        optimizer.step()

        if (
            EXPLORATION_NOISE_STD > 0.0
            and epoch < EXPLORATION_EPOCHS
            and EXPLORATION_NOISE_EVERY > 0
            and epoch % EXPLORATION_NOISE_EVERY == 0
        ):
            progress = epoch / float(max(1, EXPLORATION_EPOCHS))
            noise_std = EXPLORATION_NOISE_STD * (1.0 - progress)
            with torch.no_grad():
                for param in optim_params:
                    param.add_(noise_std * torch.randn_like(param))
                c_sin_A[:, 0] = 0.0
                c_sin_B[:, 0] = 0.0

        loss_history.append(loss.item())

        if epoch % 50 == 0:
            norm_rmse = torch.sqrt(F.mse_loss(torque_pred_norm, target_torque_norm)).item()
            torque_rmse = torch.sqrt(F.mse_loss(torque_pred, target_torque)).item()
            curve_std = torque_pred_norm.std().item()
            print(
                f"Epoch {epoch:4d} | "
                f"Loss={loss.item():.6f} | "
                f"WMSE={loss_weighted_mse.item():.6f} | "
                f"RMSE(norm)={norm_rmse:.4f} | "
                f"RMSE(Nm)={torque_rmse:.4f} | "
                f"Amp={loss_amp.item():.6f} | "
                f"dY={loss_deriv.item():.6f} | "
                f"Rad={rad_loss.item():.6f} | "
                f"Bin={bin_loss.item():.6f} | "
                f"RadW={rad_weight:.4f} | "
                f"CurveSTD(norm)={curve_std:.6f}"
            )

        should_save_frame = (epoch % gif_every == 0) or (epoch == num_epochs - 1)
        should_display = show_live_optimization and ((epoch % display_every == 0) or (epoch == num_epochs - 1))
        if should_save_frame or should_display:
            plate_A_np = pattern_A.detach().cpu().numpy()
            plate_B_np = pattern_B.detach().cpu().numpy()
            torque_pred_np = torque_pred.detach().cpu().numpy()

            if should_display:
                live_im_A.set_data(plate_A_np)
                live_im_B.set_data(plate_B_np)
                live_ax2.set_title(f"Disk B (Epoch {epoch})")
                live_pred_line.set_ydata(torque_pred_np)
                live_ax3.relim()
                live_ax3.autoscale_view(scalex=False, scaley=True)
                live_fig.tight_layout()
                live_fig.canvas.draw_idle()
                live_fig.canvas.flush_events()
                plt.pause(0.001)

            if not should_save_frame:
                continue

            fig = plt.figure(figsize=(12, 6))

            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(plate_A_np, cmap="gray", origin="lower")
            ax1.set_title("Disk A (white=PLA, black=TPU)")
            ax1.axis("off")

            ax2 = fig.add_subplot(2, 3, 2)
            ax2.imshow(plate_B_np, cmap="gray", origin="lower")
            ax2.set_title(f"Disk B (Epoch {epoch})")
            ax2.axis("off")

            ax3 = fig.add_subplot(2, 1, 2)
            ax3.plot(angle_deg_np, target_torque_np, label="Target Piecewise-Linear Torque", linewidth=2)
            ax3.plot(angle_deg_np, torque_pred_np, "--", label="Predicted Torque")
            ax3.set_title("Disk Friction Torque vs Rotation Angle")
            ax3.set_xlabel("Angle (deg)")
            ax3.set_ylabel("Torque (N.m, scaled)")
            ax3.set_xlim(0.0, 360.0)
            ax3.set_xticks(np.arange(0.0, 361.0, 45.0))
            ax3.grid(True)
            ax3.legend()

            plt.tight_layout()
            if should_save_frame:
                frame_path = frame_dir / f"frame_{epoch:05d}.png"
                fig.savefig(frame_path, dpi=150)
                frame_paths.append(frame_path)
            plt.close(fig)

    if show_live_optimization and live_fig is not None:
        plt.ioff()
        live_fig.canvas.draw_idle()
        live_fig.canvas.flush_events()
        live_fig.show()

    # Export final optimized patterns as DXF files for CAD/Fusion workflows.
    export_plate_pattern_to_dxf(
        pattern=pattern_A,
        disk_mask=disk_mask,
        out_dxf_path="optimized_plate_A_pattern.dxf",
        plate_radius_mm=PLATE_RADIUS_MM,
        pla_threshold=0.5,
    )
    export_plate_pattern_to_dxf(
        pattern=pattern_B,
        disk_mask=disk_mask,
        out_dxf_path="optimized_plate_B_pattern.dxf",
        plate_radius_mm=PLATE_RADIUS_MM,
        pla_threshold=0.5,
    )

    # Build optimization-evolution GIF from saved training frames.
    if frame_paths:
        gif_path = Path("optimization_progress_test3.gif")
        with imageio.get_writer(gif_path, mode="I", duration=gif_frame_duration_s, loop=0) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        print(f"GIF exported: {gif_path.resolve()}")
