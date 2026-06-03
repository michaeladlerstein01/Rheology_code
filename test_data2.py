import numpy as np 
import torch 
from compute_wimg import *
from matplotlib import pyplot as plt

def radial_edge_unimodality_loss(img: torch.Tensor,
                                 radius_px: int,
                                 num_angles: int = 72,
                                 num_r: int | None = None) -> torch.Tensor:
    """
    Encourage a single boundary crossing per ray from the center.
    We detect edges along radius via |dp/dr| weighted by p*(1-p),
    then minimize the entropy of that edge distribution (unimodality).
    """
    if img.ndim == 2:
        x = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    else:
        x = img
    H, W = x.shape[-2:]
    if num_r is None:
        num_r = radius_px + 1

    # Angles 0..2π (exclude endpoint)
    theta = torch.linspace(0.0, 2*math.pi, steps=num_angles+1, device=x.device)[:-1]  # (A,)
    r = torch.linspace(0.0, float(radius_px), steps=num_r, device=x.device)           # (R,)

    # Build sampling grid in normalized coordinates [-1,1]
    # Center at (0,0), normalize by radius_px so rays stay inside the disk.
    # grid: (1, A, R, 2)
    rr = r.view(1, 1, -1).expand(1, theta.numel(), -1)        # (1,A,R)
    tt = theta.view(1, -1, 1).expand(1, theta.numel(), r.numel())  # (1,A,R)

    gx = (rr * torch.cos(tt)) / radius_px   # normalized x
    gy = (rr * torch.sin(tt)) / radius_px   # normalized y
    grid = torch.stack([gx, gy], dim=-1)    # (1,A,R,2)

    # Sample along rays
    pr = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (1,1,A,R)
    pr = pr[0,0]  # (A,R), values in [0,1] due to sigmoid in generator

    # Edge strength along radius (between samples)
    dp = pr[:, 1:] - pr[:, :-1]                           # (A,R-1)
    w = 4.0 * pr * (1.0 - pr)                             # (A,R)
    w_mid = 0.5 * (w[:, 1:] + w[:, :-1])                  # (A,R-1)
    edge = (dp.abs() * w_mid)                             # (A,R-1), >=0

    eps = 1e-8
    # Normalize edge distribution along each ray and minimize entropy
    edge_sum = edge.sum(dim=1, keepdim=True) + eps        # (A,1)
    p_edge = edge / edge_sum                              # (A,R-1)
    entropy = -(p_edge * (p_edge + eps).log()).sum(dim=1) # (A,)
    # Normalize by log(R-1) so loss is in [0,1]; lower is better (more unimodal)
    entropy = entropy / math.log(max(2, edge.shape[1]))
    return entropy.mean()

if __name__ == "__main__":       

    #Generate the desired curve
    min_bound =  300
    max_bound  =  2463 #complete area overlapp for the entire plates filled in 
    x = torch.linspace(0 , 2*torch.pi , 72)
    # test_area =  800*torch.sin((x) + torch.pi/2) + 650 
    # test_area = 500*x
    triangle =  torch.abs(2 * (x / (2*torch.pi) - torch.floor(x / (2*torch.pi) + 0.5)))  
    sine = torch.sin(4*x) + 1
    # Scale to match your previous amplitude and offset


    test_area = 500 * sine
    # test_area = 2000.0 * torch.ones(72, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))    
    plt.show()

    #intitilize constancts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, N, grid_size = 4,4, 64
    beta = 10.0
    c_cos = torch.nn.Parameter(torch.randn(N + 1, M + 1, device=device))
    c_sin = torch.nn.Parameter(torch.randn(N + 1, M + 1, device=device))
    with torch.no_grad():
        c_sin[:, 0] = 0.0

    # Ground truth pattern (frozen)
    with torch.no_grad():
        # fixed_plate = generate_pattern_torch(grid_size, M, N, beta, c_cos, c_sin).detach()
        pass
    M2, N2, grid_size = 4, 4, 64
    # New random params to train
    c_cos_B = torch.nn.Parameter(torch.randn(N2 + 1, M2 + 1, device=device))
    c_sin_B = torch.nn.Parameter(torch.randn(N2 + 1, M2 + 1, device=device))
    c_cos_A = torch.nn.Parameter(torch.randn(N2 + 1, M2 + 1, device=device))
    c_sin_A = torch.nn.Parameter(torch.randn(N2 + 1, M2 + 1, device=device))
    with torch.no_grad():
        c_sin_B[:, 0] = 0.0
    optimizer = torch.optim.Adam([c_cos_B, c_sin_B , c_cos_A , c_sin_A], lr=0.01)


    loss_history = []

    for epoch in range(10000):
        optimizer.zero_grad()
        #generate a starting plate for B
        pattern_A = generate_pattern_torch(grid_size, M2, N2, beta, c_cos_A, c_sin_A)
        pattern_B = generate_pattern_torch(grid_size, M2, N2, beta, c_cos_B, c_sin_B)
        area_pred = area_sweep_torch(pattern_A, pattern_B)

        # include a penalty if the areas are disconnected between each other 
        
        
        loss_mse = F.mse_loss(area_pred, test_area)

        radius_px = grid_size // 2 - 4

        rad_loss_B = radial_edge_unimodality_loss(pattern_B, radius_px= radius_px , num_angles= 72)
        rad_loss_A = radial_edge_unimodality_loss(pattern_A , radius_px= radius_px , num_angles= 72)
        rad_loss =    100 * (rad_loss_B + rad_loss_A)
        loss = loss_mse +  rad_loss

        loss.backward()
        with torch.no_grad():
            c_sin_B.grad[:, 0] = 0.0
        optimizer.step()
        loss_history.append(loss.item()) 

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss_mse = {loss_mse.item():.4f} , radial_loss = {rad_loss.item():.4f} ")
        
        if epoch % 2000 == 0:
            pattern_B_plt = pattern_B.detach().cpu().numpy()
            pattern_A_plt = pattern_A.detach().cpu().numpy()
            area_plt = area_pred.detach().cpu().numpy()
            test_area_np = test_area.detach().cpu().numpy()
            
            # Create a figure with 2 rows: 2 image plots and 1 curve plot
            fig = plt.figure(figsize=(12, 6))

            # Plot fixed plate
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(pattern_A_plt, cmap='gray', origin='lower')
            ax1.set_title("Fixed Plate (A)")
            ax1.axis('off')

            # Plot current optimized plate
            ax2 = fig.add_subplot(2, 3, 2)
            ax2.imshow(pattern_B_plt, cmap='gray', origin='lower')
            ax2.set_title(f"Optimized Plate B (Epoch {epoch})")
            ax2.axis('off')

            # Plot overlap curves
            ax3 = fig.add_subplot(2, 1, 2)
            ax3.plot(x , test_area_np, label='Target Curve', linewidth=2)
            ax3.plot(x , area_plt, label='Predicted Curve', linestyle='--')
            ax3.set_title("Overlap Area vs. Rotation Angle")
            ax3.set_xlabel("Angle (degrees)")
            ax3.set_ylabel("Overlap Area")
            ax3.grid(True)
            ax3.legend()

            plt.tight_layout()
            plt.show()



      
            






    
