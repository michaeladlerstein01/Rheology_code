from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANGLE_COLUMN_CANDIDATES = (
    "python_angle_deg",
    "angle_deg",
    "rotation_deg",
    "angle",
    "ur3_deg",
)
TORQUE_COLUMN_CANDIDATES = (
    "friction_torque_nm",
    "resisting_torque_nm",
    "signed_resisting_torque_nm",
    "reaction_moment_nm",
    "torque_nm",
    "moment_nm",
    "friction_torque_nmm",
    "resisting_torque_nmm",
    "signed_resisting_torque_nmm",
    "reaction_moment_nmm",
)


def _normalized_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _choose_column(headers: list[str], candidates: tuple[str, ...], label: str) -> str:
    by_normalized = {_normalized_name(header): header for header in headers}
    for candidate in candidates:
        if candidate in by_normalized:
            return by_normalized[candidate]

    joined = ", ".join(headers)
    raise ValueError(f"Could not find a {label} column. Available columns: {joined}")


def resolve_csv_path(csv_arg: str) -> Path:
    requested = Path(csv_arg)
    if requested.exists():
        return requested

    for candidate in ("torque_vs_angle.csv", "torque v angle.csv", "torque_v_angle.csv"):
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(f"Could not find {csv_arg} or any default torque-angle CSV file.")


def load_abaqus_torque_csv(
    csv_path: Path,
    angle_column: str | None,
    torque_column: str | None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row.")

        headers = reader.fieldnames
        angle_col = angle_column or _choose_column(headers, ANGLE_COLUMN_CANDIDATES, "angle")
        torque_col = torque_column or _choose_column(headers, TORQUE_COLUMN_CANDIDATES, "torque")

        angles = []
        torques = []
        for row in reader:
            try:
                angle = float(row[angle_col])
                torque = float(row[torque_col])
            except (TypeError, ValueError):
                continue

            if _normalized_name(torque_col).endswith("_nmm"):
                torque *= 1e-3

            angles.append(angle)
            torques.append(torque)

    angle_np = np.asarray(angles, dtype=float)
    torque_np = np.asarray(torques, dtype=float)
    keep = np.isfinite(angle_np) & np.isfinite(torque_np)
    angle_np = angle_np[keep]
    torque_np = torque_np[keep]

    order = np.argsort(angle_np)
    return angle_np[order], torque_np[order], angle_col, torque_col


def load_saved_optimization_curves(curve_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not curve_path.exists():
        raise FileNotFoundError(
            f"Could not find {curve_path}. Run test_data3.py first so it saves "
            "results/test3_final/optimized_torque_curves.npz."
        )

    data = np.load(curve_path)
    required = ("angle_deg", "target_torque_nm", "optimized_torque_nm")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{curve_path} is missing required arrays: {', '.join(missing)}")

    return (
        np.asarray(data["angle_deg"], dtype=float),
        np.asarray(data["target_torque_nm"], dtype=float),
        np.asarray(data["optimized_torque_nm"], dtype=float),
    )


def save_processed_overlay_csv(
    out_path: Path,
    opt_angle: np.ndarray,
    target_torque: np.ndarray,
    optimized_torque: np.ndarray,
    abaqus_angle: np.ndarray,
    abaqus_torque: np.ndarray,
) -> None:
    abaqus_interp = np.interp(opt_angle, abaqus_angle, abaqus_torque, left=np.nan, right=np.nan)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "target_torque_nm", "optimized_torque_nm", "abaqus_torque_nm_interpolated"])
        for row in zip(opt_angle, target_torque, optimized_torque, abaqus_interp):
            writer.writerow([float(row[0]), float(row[1]), float(row[2]), float(row[3])])


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay Abaqus torque-angle data with saved Python optimization curves.")
    parser.add_argument("--csv", default="torque_vs_angle.csv", help="Abaqus torque-angle CSV file.")
    parser.add_argument(
        "--curves",
        default="results/test3_final/optimized_torque_curves.npz",
        help="Saved curves from test_data3.py.",
    )
    parser.add_argument("--angle-column", default=None, help="CSV angle column override.")
    parser.add_argument("--torque-column", default=None, help="CSV torque column override.")
    parser.add_argument("--output", default="torque_angle_comparison.png", help="Output plot PNG path.")
    parser.add_argument("--processed-csv", default="torque_angle_comparison_processed.csv", help="Output processed overlay CSV path.")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv)
    curves_path = Path(args.curves)
    output_path = Path(args.output)
    processed_csv_path = Path(args.processed_csv)

    abaqus_angle, abaqus_torque, angle_col, torque_col = load_abaqus_torque_csv(
        csv_path=csv_path,
        angle_column=args.angle_column,
        torque_column=args.torque_column,
    )
    opt_angle, target_torque, optimized_torque = load_saved_optimization_curves(curves_path)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(opt_angle, target_torque, linewidth=2.5, label="Saved Python target")
    ax.plot(opt_angle, optimized_torque, "--", linewidth=2.2, label="Saved Python optimized curve")
    ax.plot(abaqus_angle, abaqus_torque, linewidth=1.8, alpha=0.85, label="Abaqus extracted torque")
    ax.set_title("Torque vs Rotation Angle")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Torque (N.m)")
    ax.set_xlim(0.0, 360.0)
    ax.set_xticks(np.arange(0.0, 361.0, 45.0))
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    save_processed_overlay_csv(
        out_path=processed_csv_path,
        opt_angle=opt_angle,
        target_torque=target_torque,
        optimized_torque=optimized_torque,
        abaqus_angle=abaqus_angle,
        abaqus_torque=abaqus_torque,
    )

    print(f"Used Abaqus CSV: {csv_path.resolve()}")
    print(f"Used Abaqus columns: angle='{angle_col}', torque='{torque_col}'")
    print(f"Used saved optimization curves: {curves_path.resolve()}")
    print(f"Saved plot: {output_path.resolve()}")
    print(f"Saved processed overlay CSV: {processed_csv_path.resolve()}")


if __name__ == "__main__":
    main()
