
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def show_training_viz(epoch, mat1, mat2, shifts, out_curve, ref_curve):
    mat1_cpu = mat1.detach().float().cpu()
    mat2_cpu = mat2.detach().float().cpu()
    out_cpu  = out_curve.detach().float().cpu()
    ref_cpu  = ref_curve.detach().float().cpu()
    shifts_cpu = shifts.detach().cpu()

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    # Plates
    im0 = axes[0,0].imshow(mat1_cpu, origin="lower", interpolation="nearest",
                           extent=[0,1,0,1], aspect="auto", vmin=0, vmax=1)
    axes[0,0].set_title(f"Plate 1 (soft) — epoch {epoch}")
    axes[0,0].set_xticks([]); axes[0,0].set_yticks([])
    fig.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)

    im1 = axes[0,1].imshow(mat2_cpu, origin="lower", interpolation="nearest",
                           extent=[0,1,0,1], aspect="auto", vmin=0, vmax=1)
    axes[0,1].set_title("Plate 2 (soft / fixed)")
    axes[0,1].set_xticks([]); axes[0,1].set_yticks([])
    fig.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)

    # Overlap curve vs reference
    axes[1,0].plot(shifts_cpu, out_cpu, label="overlap (norm)")
    axes[1,0].plot(shifts_cpu, ref_cpu, label="reference", linestyle="--")
    axes[1,0].set_xlabel("shift (columns)")
    axes[1,0].set_ylabel("normalized overlap")
    axes[1,0].set_title("Overlap vs Reference")
    axes[1,0].legend()

    # Empty (or you can add extra info)
    axes[1,1].axis("off")

    plt.tight_layout()
    plt.show()
# ---------- Bases & params ----------

def build_fourier_bases(H, W, Mx, My, device=None, dtype=torch.float32):
    """
    Precompute cosine/sine bases over an HxW grid with normalized coords in [0,1].
    Returns a dict of tensors suitable for reuse across calls.
    """
    device = device or torch.device("cpu")
    y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    twopiX = 2 * math.pi * X
    twopiY = 2 * math.pi * Y

    cosX = torch.nn.Parameter(torch.stack([torch.cos(m * twopiX) for m in range(Mx + 1)], dim=0))  # (Mx+1, H, W)
    sinX = torch.nn.Parameter(torch.stack([torch.sin(m * twopiX) for m in range(Mx + 1)], dim=0))
    cosY = torch.nn.Parameter(torch.stack([torch.cos(n * twopiY) for n in range(My + 1)], dim=0)) # (My+1, H, W)
    sinY = torch.nn.Parameter(torch.stack([torch.sin(n * twopiY) for n in range(My + 1)], dim=0))

    bases = {
        "H": H, "W": W, "Mx": Mx, "My": My,
        "cosX": cosX, "sinX": sinX, "cosY": cosY, "sinY": sinY,
        "device": device, "dtype": dtype,
    }
    return bases , cosX , sinX , cosY , sinY


def init_fourier_params(Mx, My, init_scale=0.1, device=None, dtype=torch.float32):
    """
    Initialize learnable Fourier coefficients and global bias tau.
    Returns a dict of nn.Parameter tensors so you can optimize them directly.
    """
    device = device or torch.device("cpu")
    shape = (Mx + 1, My + 1)
    params = {
        "A_cc": nn.Parameter(init_scale * torch.randn(*shape, device=device, dtype=dtype)),
        "A_cs": nn.Parameter(init_scale * torch.randn(*shape, device=device, dtype=dtype)),
        "A_sc": nn.Parameter(init_scale * torch.randn(*shape, device=device, dtype=dtype)),
        "A_ss": nn.Parameter(init_scale * torch.randn(*shape, device=device, dtype=dtype)),
        "tau":  nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype)),
    }
    return params


def params_list(params_dict):
    """Convenience: get a list of parameters for an optimizer."""
    return [params_dict[k] for k in ("A_cc", "A_cs", "A_sc", "A_ss", "tau")]


# ---------- Field & forward ----------

def fourier_field(params, bases):
    """
    Compute F(x,y) = Σ_{m,n} [ A_cc cos(mx)cos(ny) + A_cs cos(mx)sin(ny)
                              + A_sc sin(mx)cos(ny) + A_ss sin(mx)sin(ny) ].
    Vectorized over (m,n) for speed; returns (H, W).
    """
    cosX, sinX, cosY, sinY = bases["cosX"], bases["sinX"], bases["cosY"], bases["sinY"]

    # Shape alignments:
    # cosX: (Mx+1, H, W) -> (Mx+1, 1, H, W)
    # cosY: (My+1, H, W) -> (1, My+1, H, W)
    CX = cosX[:, None, ...]
    SX = sinX[:, None, ...]
    CY = cosY[None, :, ...]
    SY = sinY[None, :, ...]

    # Build the four basis stacks (Mx+1, My+1, H, W)
    B_cc = CX * CY
    B_cs = CX * SY
    B_sc = SX * CY
    B_ss = SX * SY

    # Contract (m,n) with coefficients -> (H, W)
    F_cc = torch.einsum("mn,mnhw->hw", params["A_cc"], B_cc)
    F_cs = torch.einsum("mn,mnhw->hw", params["A_cs"], B_cs)
    F_sc = torch.einsum("mn,mnhw->hw", params["A_sc"], B_sc)
    F_ss = torch.einsum("mn,mnhw->hw", params["A_ss"], B_ss)
    F_total = F_cc + F_cs + F_sc + F_ss
    return F_total


def fourier_plate(params, bases, beta=10.0, hard=False):
    """
    Differentiable occupancy in (0,1) via sigmoid(beta * (F - tau)).
    If hard=True, returns straight-through hard mask (0/1) with gradients from the soft map.
    """
    F_xy = fourier_field(params, bases)
    p = torch.sigmoid(beta * (F_xy - params["tau"]))
    if not hard:
        return p
    hard_mask = (p > 0.5).float()
    return hard_mask + (p - p.detach())


# ---------- Pretty plotting ----------
def show_plates_grid(plates, titles=None, ncols=2, figsize=(10, 5)):
    """
    Display multiple plates in a grid of subplots with shared colorbar.
    Args:
        plates  : list of (H,W) tensors
        titles  : list of strings, optional
        ncols   : number of columns in the subplot grid
        figsize : overall figure size
    """
    n = len(plates)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    vmin, vmax = 0.0, 1.0   # consistent scale
    ims = []

    for i, p in enumerate(plates):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        im = ax.imshow(
            p.detach().float().cpu(),
            origin="lower",
            interpolation="nearest",
            extent=[0, 1, 0, 1],
            aspect="auto",
            vmin=vmin, vmax=vmax,
        )
        ims.append(im)
        if titles:
            ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplots
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    # Shared colorbar
    cbar = fig.colorbar(ims[0], ax=axes, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("occupancy", rotation=270, labelpad=12)

    plt.tight_layout()
    plt.show()


def calculate_overlap(mat1 , mat2 , pos):
    #mat2 is longer than mat1 
    # slice mat2 to have the same dimension as mat1 
    dims = mat1.size()
    slice = mat2[: ,  pos: mat1.size()[1]+pos]
    overlap_area = torch.sum(mat1 * slice)
    return overlap_area

    

# ---------- Example usage ----------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    pos = 0 
    H, W = 20, 120
    H1, W1 =  20 , 200 # 20 , 200
    Mx, My = 3, 3
    beta = 3.0
    max_slide = W1 - W
    eps = 1e-10
    x_axis = torch.linspace(0 , max_slide-1, max_slide)
    # ref = torch.sigmoid(1/10*x_axis - 20)
    # ref = torch.ones(500) * 0.7

    # Build constant bases (no grad)
    bases1 = build_fourier_bases(H, W, Mx, My, device=device, dtype=dtype)  # dict, constants
    bases2 = build_fourier_bases(H1, W1, Mx, My, device=device, dtype=dtype)

    # Learnable params (nn.Parameter) – NOT the bases
    plate1_params = init_fourier_params(Mx, My, init_scale=0.1, device=device, dtype=dtype)
    # If mat2 is a fixed reference, precompute and detach it:
    
    # IF YOU WANT TO OPTIMIZE ONLY ONE PLATE WITH THE OTHER FIXED
    # with torch.no_grad():
    #     plate2_params = init_fourier_params(Mx, My, device=device, dtype=dtype)
    #     mat2_fixed = fourier_plate(plate2_params,  # or load your target
    #                             bases2[0], beta=beta, hard=True)
    # mat2_fixed = mat2_fixed.detach()  # ensures no grads flow through mat2

    plate2_params = init_fourier_params(Mx, My, device=device, dtype=dtype)

    optimizer = torch.optim.Adam(
        params_list(plate1_params) + params_list(plate2_params),
        lr=1.2
    )

    losses = []
    num_shifts = max_slide - pos
    x_axis = torch.linspace(0, max_slide - 1, max_slide, device=device, dtype=dtype)
    ref = torch.sigmoid(0.1 * x_axis - 20)[pos:max_slide] * 0.4 + 0.2 # match shape (num_shifts,)


    # ref = torch.sigmoid(0.03 * x_axis - 2) * 0.4 + 0.2
    ref = torch.sigmoid(0.1 * x_axis - 4) * 0.4 + 0.2
    # ref = 0.3 * (torch.sin(x_axis/ 10) + 2 )
    # ref = torch.ones(500) * 0.4

    Hc = min(H , H1)
    Wc = min(W , W1)  # max possible overlap width
    C  = float(Hc * Wc)  # fixed scale
    disp = True

    shifts = torch.arange(pos, max_slide, device=device, dtype=dtype)  # for x-axis
    plt.ion()
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 6),
        gridspec_kw={'width_ratios': [W, W1]}  # plate2 axis is 2× wider than plate1
    )
        # --- Training ---
    for epoch in range(5000):
        optimizer.zero_grad()

        # Recompute mat1 from CURRENT params each epoch
        mat1 = fourier_plate(plate1_params, bases1[0], beta=beta, hard=False)
        mat2 = fourier_plate(plate2_params, bases2[0], beta=beta, hard=False)

        # Collect overlaps as a fresh tensor (no accumulation across epochs)
        areas = torch.empty(num_shifts, device=device, dtype=dtype)
        for i, shift in enumerate(range(pos, max_slide)):
            areas[i] = calculate_overlap(mat1, mat2, shift)

        # Normalize to [0,1] safely
        a_min, a_max = areas.min(), areas.max()
        # out = (areas - a_min) / (a_max - a_min + eps)
        out = areas / (C + eps)
        
        loss = F.mse_loss(out, ref)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())

        if epoch % 10 == 0:
            print(f"loss {loss.item():.6f}")
            if disp is True:
                for ax in axes.ravel():
                    ax.clear()

                im0 = axes[0,0].imshow(
                    mat1.detach().cpu(),
                    origin="lower",
                    interpolation="nearest",
                    extent=[0, W, 0, H],
                    aspect="equal",   # <-- keep aspect ratio true to data
                    vmin=0, vmax=1
                )

                # plate 2 (width = W1, height = H1)
                im1 = axes[0,1].imshow(
                    mat2.detach().cpu(),
                    origin="lower",
                    interpolation="nearest",
                    extent=[0, W1, 0, H1],
                    aspect="equal",   # <-- keep aspect ratio true to data
                    vmin=0, vmax=1
                )
                axes[0,1].set_title("Plate 2 (soft)")
                axes[0,1].set_xticks([]); axes[0,1].set_yticks([])
                # fig.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)

                # overlap vs reference (bottom-left)
                axes[1,0].plot(shifts.cpu(), out.detach().cpu(), label="overlap (norm)")
                axes[1,0].plot(shifts.cpu(), ref.detach().cpu(), "--", label="reference")
                axes[1,0].set_xlabel("shift (columns)")
                axes[1,0].set_ylabel("overlap")
                axes[1,0].legend()

                # bottom-right empty (or put loss curve/live metrics here)
                axes[1,1].axis("off")

                fig.canvas.draw()
                fig.canvas.flush_events()

    plt.ioff()   # turn off interactive mode at the end
    plt.show()   # show final figure
    # Optional final loss plot
    plt.figure(figsize=(6,3))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.show()


    torch.save(mat1.detach().cpu(), "mat1.pt")
    torch.save(mat2.detach().cpu(), "mat2.pt")
    plt.imsave("plate1.jpg", mat1.detach().cpu().numpy(), origin="lower", vmin=0, vmax=1, cmap="viridis")
    plt.imsave("plate2.jpg", mat2.detach().cpu().numpy(), origin="lower", vmin=0, vmax=1, cmap="viridis")