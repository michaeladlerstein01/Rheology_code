#!/usr/bin/env python3
import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# ----------------- utils -----------------
def to01(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    if x.numel() == 0:
        return x
    mn, mx = x.min(), x.max()
    if (mx - mn) > 0:
        return (x - mn) / (mx - mn)
    return torch.zeros_like(x)

def calculate_overlap(mat1: torch.Tensor, mat2: torch.Tensor, pos: int):
    """
    Horizontal sliding overlap (mat1 over mat2 shifted by `pos` columns).
    Returns (overlap_window_from_mat2, product, scalar_sum).
    Works for arbitrary heights/widths; aligns at top (row 0).
    """
    device = mat1.device
    dtype  = mat1.dtype

    H1, W1 = mat1.shape[-2], mat1.shape[-1]
    H2, W2 = mat2.shape[-2], mat2.shape[-1]
    H = min(H1, H2)
    if H <= 0:
        return (mat2.new_zeros((0,0)), mat2.new_zeros((0,0)), mat2.new_tensor(0.0))

    start1 = max(0, -pos)
    start2 = max(0,  pos)
    max_w1 = W1 - start1
    max_w2 = W2 - start2
    overlap_w = min(max_w1, max_w2)
    if overlap_w <= 0:
        return (mat2.new_zeros((H,0)), mat2.new_zeros((H,0)), mat2.new_tensor(0.0))

    m1 = mat1[:H, start1:start1+overlap_w]
    m2 = mat2[:H, start2:start2+overlap_w]
    prod = m1 * m2
    return (m2, prod, prod.sum())

def load_tensor(path: str, device: torch.device):
    if path.endswith((".npy", ".npz")):
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # take the first array
            key = list(arr.keys())[0]
            arr = arr[key]
        t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    else:
        t = torch.load(path, map_location=device)
        if isinstance(t, dict):  # try common keys
            for k in ("mat", "mat1", "mat2", "tensor"):
                if k in t:
                    t = t[k]
                    break
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, device=device, dtype=torch.float32)
        else:
            t = t.to(device=device, dtype=torch.float32)
    assert t.ndim == 2, f"Expected 2D tensor, got shape {tuple(t.shape)}"
    return t

def example_field(H, W, device):
    """Nice-looking synthetic field in [0,1] for demo."""
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    f = (
        0.4*torch.cos(2*math.pi*X) * torch.cos(1*math.pi*Y)
      + 0.3*torch.sin(3*math.pi*X) * torch.sin(2*math.pi*Y)
      + 0.2*torch.cos(4*math.pi*X + 0.7) * torch.sin(3*math.pi*Y + 0.3)
    )
    return to01(f)

# ----------------- GUI -----------------
class SlideViewer:
    def __init__(self, mat1, mat2, start_pos=0):
        """
        mat1: (H1, W1) tensor in [0,1]
        mat2: (H2, W2) tensor in [0,1]
        """
        self.mat1 = mat1.detach().float()
        self.mat2 = mat2.detach().float()
        self.device = self.mat1.device

        self.H1, self.W1 = self.mat1.shape
        self.H2, self.W2 = self.mat2.shape

        # Allowed shift range so an overlap exists at least at one column
        self.min_shift = -self.W1 + 1
        self.max_shift = self.W2 - 1
        # If mat2 is wider than mat1 (common), typical useful range is [0, W2 - W1]
        # but we allow the full logical range so you can slide fully off either side:
        # overlap width becomes 0 when outside.

        self.pos = int(start_pos)
        self.pos = max(self.min_shift, min(self.max_shift, self.pos))

        # Figure layout: 2 rows x 3 cols (mat1, mat2 slice, product)
        self.fig, self.axes = plt.subplots(
            2, 3, figsize=(12, 6),
            gridspec_kw={"width_ratios": [self.W1, max(self.W1, self.W2), self.W1]}
        )
        self.fig.canvas.manager.set_window_title("Sliding Overlap Viewer")

        # top row images
        self.ax_m1   = self.axes[0,0]
        self.ax_m2   = self.axes[0,1]
        self.ax_prod = self.axes[0,2]

        # bottom row: text/status and overlap curve placeholder
        self.ax_info = self.axes[1,0]
        self.ax_curve = self.axes[1,1]
        self.axes[1,2].axis("off")

        self.im1 = self.ax_m1.imshow(self.mat1.cpu().numpy(),
                                     origin="lower", extent=[0, self.W1, 0, self.H1],
                                     aspect="auto", vmin=0, vmax=1, interpolation="nearest")
        self.ax_m1.set_title("mat1")
        self.ax_m1.set_xticks([]); self.ax_m1.set_yticks([])

        # placeholders for mat2 slice and product
        self.im2 = None
        self.im_prod = None

        # draw full mat2 as background with a rectangle showing the current slice
        self.ax_m2.imshow(self.mat2.cpu().numpy(),
                          origin="lower", extent=[0, self.W2, 0, self.H2],
                          aspect="auto", vmin=0, vmax=1, interpolation="nearest")
        self.slice_rect = self.ax_m2.add_patch(
            plt.Rectangle((0, 0), self.W1, min(self.H1, self.H2),
                          fill=False, ec="white", lw=2)
        )
        self.ax_m2.set_title("mat2 (full) + current slice")
        self.ax_m2.set_xticks([]); self.ax_m2.set_yticks([])

        self.ax_prod.set_title("elementwise product (overlap)")
        self.ax_prod.set_xticks([]); self.ax_prod.set_yticks([])

        # status text
        self.ax_info.axis("off")
        self.text = self.ax_info.text(
            0.02, 0.9, "", transform=self.ax_info.transAxes, fontsize=12, va="top"
        )

        # precompute an overlap curve across all shifts (for the bottom plot)
        shifts = torch.arange(self.min_shift, self.max_shift + 1, device=self.device)
        vals = []
        for s in shifts.tolist():
            _, _, v = calculate_overlap(self.mat1, self.mat2, s)
            vals.append(v.detach().cpu())
        self.shifts_axis = shifts.cpu().numpy()
        self.overlap_curve = torch.stack(vals).numpy()
        self.line_curve, = self.ax_curve.plot(self.shifts_axis, self.overlap_curve, label="overlap sum")
        self.line_marker = self.ax_curve.axvline(self.pos, color="r", linestyle="--", label="current pos")
        self.ax_curve.set_xlabel("shift")
        self.ax_curve.set_ylabel("overlap sum")
        self.ax_curve.legend(loc="best")

        # initial draw
        self.refresh_images()

        # key bindings
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        print(
            "Controls: 'a' → shift +1, 'd' → shift -1, "
            "'r' → reset, 'q' → quit, 's' → save PNG"
        )

        plt.tight_layout()
        plt.show()

    def refresh_images(self):
        m2slice, prod, s = calculate_overlap(self.mat1, self.mat2, self.pos)

        # update mat2 slice image
        if self.im2 is None:
            self.im2 = self.ax_m2.imshow(
                m2slice.cpu().numpy(),
                origin="lower",
                extent=[self.pos, self.pos + m2slice.shape[1], 0, m2slice.shape[0]],
                aspect="auto", vmin=0, vmax=1, interpolation="nearest", alpha=0.85
            )
        else:
            self.im2.set_data(m2slice.cpu().numpy())
            self.im2.set_extent([self.pos, self.pos + m2slice.shape[1], 0, m2slice.shape[0]])

        # update slice rectangle on mat2
        self.slice_rect.set_x(self.pos)
        self.slice_rect.set_y(0)
        self.slice_rect.set_width(self.W1)
        self.slice_rect.set_height(min(self.H1, self.H2))

        # update product image
        if self.im_prod is None:
            self.im_prod = self.ax_prod.imshow(
                prod.cpu().numpy(),
                origin="lower", extent=[0, prod.shape[1], 0, prod.shape[0]],
                aspect="auto", vmin=0, vmax=1, interpolation="nearest"
            )
        else:
            self.im_prod.set_data(prod.cpu().numpy())
            self.im_prod.set_extent([0, prod.shape[1], 0, prod.shape[0]])

        # update status text and current pos marker
        self.text.set_text(f"shift = {self.pos}\noverlap sum = {float(s):.4f}")
        self.line_marker.set_xdata(self.pos)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def on_key(self, event):
        if event.key == "a":        # shift to the RIGHT by +1
            self.pos = min(self.pos + 1, self.max_shift)
            self.refresh_images()
        elif event.key == "d":      # shift to the LEFT by -1
            self.pos = max(self.pos - 1, self.min_shift)
            self.refresh_images()
        elif event.key == "r":
            self.pos = 0
            self.refresh_images()
        elif event.key == "s":
            self.fig.savefig("slide_view.png", dpi=150, bbox_inches="tight")
            print("Saved slide_view.png")
        elif event.key == "q":
            plt.close(self.fig)

# ----------------- main -----------------
def main():
    parser = argparse.ArgumentParser(description="GUI to slide two matrices and inspect overlap.")
    parser.add_argument("--mat1", type=str, default=None, help="path to mat1 (.pt/.pth/.npy)")
    parser.add_argument("--mat2", type=str, default=None, help="path to mat2 (.pt/.pth/.npy)")
    parser.add_argument("--H", type=int, default=100, help="demo height for generated mat1")
    parser.add_argument("--W", type=int, default=500, help="demo width  for generated mat1")
    parser.add_argument("--H2", type=int, default=100, help="demo height for generated mat2")
    parser.add_argument("--W2", type=int, default=1000, help="demo width  for generated mat2")
    parser.add_argument("--start", type=int, default=0, help="starting shift")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mat1 is not None:
        mat1 = load_tensor(args.mat1, device)
        mat1 = to01(mat1)
    else:
        mat1 = example_field(args.H, args.W, device)

    if args.mat2 is not None:
        mat2 = load_tensor(args.mat2, device)
        mat2 = to01(mat2)
    else:
        mat2 = example_field(args.H2, args.W2, device).roll(shifts=20, dims=1)  # slight offset for visual interest

    # Launch GUI
    SlideViewer(mat1, mat2, start_pos=args.start)

if __name__ == "__main__":
    plt.ion()  # smoother refresh
    main()