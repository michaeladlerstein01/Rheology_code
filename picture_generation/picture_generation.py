import torch
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
import os
import math
import torch.nn.functional as F

#1  - import 2 pictures 
#2 -  make them the same size and square and black and white bolean 200 x 200 
#3 - convert the two images to a torch tensor 
#4 - generate two matrices initialized with random values of infill (0 black - 1 white)
#5 - define a Inplicit representation with a fully connected neural network 
#6 - rotate the two plates past each other at angle 0 match the intersection of the two areas to the first picture
#7 - at 180 match the loss to the second picture
#8 - perform gradient descent to optimize the two losses      
# 
#
# 

def process_image(path):
    # Open the image
    img = Image.open(path)
    # Resize to 200x200
    img = img.resize((200, 200))
    # Convert to grayscale
    img = img.convert("L")  # "L" mode is (8-bit pixels, black and white)
    # Convert to binary: threshold at 128
    # img = img.point(lambda x: 255 if x > 128 else 0, mode='1')
    return img


def image_to_tensor(img):
    # Convert PIL Image to NumPy array
    arr = np.array(img)  # shape: (H, W)
    
    # Normalize to [0, 1] and convert to tensor
    tensor = torch.tensor(arr, dtype=torch.float32) / 255.0
    
    # Add channel and batch dimensions: (1, 1, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    return tensor

def rot180(t):
    """Rotate a 4-D tensor (N,C,H,W) by 180° (flip H and W)."""
    return torch.flip(t, dims=[-2, -1])


def rot90(t, clockwise=False):
    k = 3 if clockwise else 1          # k=1 = +90° CCW, k=3 = −90° (270°) CW
    return torch.rot90(t, k=k, dims=(-2, -1))




def rotate_tensor(t1 , angle_deg):
    angle_rad = -angle_deg * math.pi / 180
    theta = torch.tensor([
        [math.cos(angle_rad), -math.sin(angle_rad), 0.0],
        [math.sin(angle_rad),  math.cos(angle_rad), 0.0]
    ], dtype=torch.float32, device=t1.device).unsqueeze(0)
    grid = F.affine_grid(theta, t1.size(), align_corners=False)
    return F.grid_sample(t1, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

def sigmoid_compression(p1 , p2):          # convenience – always return squashed versions
    M1 = torch.sigmoid(p1)
    M2 = torch.sigmoid(p2)
    return M1, M2


def load_grayscale_image(path, size=100):
    """
    Load an image, resize to `size`×`size`, convert to grayscale,
    and return a tensor of shape (1, 1, H, W) with values in [0, 1].

    No thresholding or binarisation is applied.
    """
    img = Image.open(path).convert("L").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0           # scale to 0-1
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)     # (1, 1, H, W)
    return t

def load_binary_image(path, size=200, thresh=128):
    """Load image, make 200×200, convert to binary 0/1 tensor of shape (1,1,H,W)."""
    img = Image.open(path).convert("L").resize((size, size))
    binary = (np.array(img) > thresh).astype(np.float32)          # 0 / 1
    t = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)        # (1,1,H,W)
    return t

def save_img(transparency_np, filename):
    scale   = 12          # factor (100×100 → 1200×1200)
    out_px  = (W*scale, H*scale)

    """Save learned transparency map as a grayscale ink plate."""
    ink = (transparency_np) * 255.0            # 0-255 ink density
    ink = ink.astype(np.uint8)
    plate_img = Image.fromarray(ink, mode="L")       # 'L' = 8-bit greyscale
    if scale > 1:
        plate_img = plate_img.resize(out_px, Image.BILINEAR)
    plate_img.save(filename, dpi=(300, 300))


def tv(x):
    return (x[..., :-1, :] - x[..., 1:, :]).abs().mean() + \
           (x[..., :, :-1] - x[..., :, 1:]).abs().mean()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size   = 300
    H, W   = size, size

    # 1. target transparencies (0=black ink, 1=clear)
    img1 = load_grayscale_image("picture_generation/franco.jpeg",   size).to(device)
    img2 = load_grayscale_image("picture_generation/dali.jpg", size).to(device)
    img3 = load_grayscale_image("picture_generation/pedro.jpg",  size).to(device)
    img4 = load_grayscale_image("picture_generation/santi.jpeg",  size).to(device)



    # 2. learnable masks
    param1 = torch.nn.Parameter(torch.randn(1, 1, H, W, device=device))
    param2 = torch.nn.Parameter(torch.randn(1, 1, H, W, device=device))

    opt, steps = torch.optim.Adam([param1, param2], lr=1e-2), 10000

    for step in range(steps):
        M1, M2 = sigmoid_compression(param1, param2)
        # M1, M2 =param1, param2

        # 3. optical products at 0°, 90°, 180°
        pred0   = M1 * M2
        pred90  = M1 * rot90(M2 , clockwise = False)       # +90 ° CCW
        pred180 = M1 * rot180(M2)
        pred270 = M1 * rot90(M2 , clockwise = True)

        loss0   = F.mse_loss(pred0,   img1)
        loss90  = F.mse_loss(pred90,  img3)
        loss180 = F.mse_loss(pred180, img2)
        loss270 = F.mse_loss(pred270, img4)

        λ = 1e-3
        loss    =  1.5 * loss0  + loss90 + loss180 +  loss270 + λ*(tv(M1)+tv(M2))

        opt.zero_grad();  loss.backward();  opt.step()

        if step % 200 == 0 or step == steps-1:
            print(f"step {step:5d} | loss={loss:.6f} "
                  f"(0°={loss0:.4f}, 90°={loss90:.4f}, 180°={loss180:.4f})")


    # ── visual inspection -------------------------------------------------------
    with torch.no_grad():
        M1, M2 = sigmoid_compression(param1, param2)
        out0   = (M1 * M2).squeeze().cpu().numpy()
        out90  = (M1 * rot90(M2)).squeeze().cpu().numpy()
        out180 = (M1 * rot180(M2)).squeeze().cpu().numpy()
        out270 = (M1 * rot90(M2, clockwise=True)).squeeze().cpu().numpy()

    fig, axs = plt.subplots(4, 3, figsize=(9, 12))
    # 0°
    axs[0,0].imshow(img1.squeeze().cpu(), cmap="gray");  axs[0,0].set_title("target 0°")
    axs[0,1].imshow(out0,                  cmap="gray");  axs[0,1].set_title("fit 0°")
    axs[0,2].imshow((img1.squeeze()-out0)**2, cmap="gray");  axs[0,2].set_title("error 0°")
    # 90°
    axs[1,0].imshow(img3.squeeze().cpu(), cmap="gray");  axs[1,0].set_title("target 90°")
    axs[1,1].imshow(out90,                 cmap="gray");  axs[1,1].set_title("fit 90°")
    axs[1,2].imshow((img3.squeeze()-out90)**2, cmap="gray"); axs[1,2].set_title("error 90°")
    # 180°
    axs[2,0].imshow(img2.squeeze().cpu(), cmap="gray");  axs[2,0].set_title("target 180°")
    axs[2,1].imshow(out180,                cmap="gray");  axs[2,1].set_title("fit 180°")
    axs[2,2].imshow((img2.squeeze()-out180)**2, cmap="gray"); axs[2,2].set_title("error 180°")
    # 270°
    axs[3,0].imshow(img4.squeeze().cpu(), cmap="gray");  axs[3,0].set_title("target 270°")
    axs[3,1].imshow(out270,                cmap="gray");  axs[3,1].set_title("fit 270°")
    axs[3,2].imshow((img4.squeeze()-out270)**2, cmap="gray"); axs[3,2].set_title("error 270°")

    for ax in axs.ravel(): ax.axis("off")
    plt.tight_layout(); plt.show()

    # ── product previews --------------------------------------------------------
    M1_img, M2_img = M1.squeeze().cpu().numpy(), M2.squeeze().cpu().numpy()
    combined_0   = np.clip(M1_img * M2_img,                       0, 1)
    combined_90  = np.clip(M1_img * np.rot90(M2_img, 1),          0, 1)
    combined_180 = np.clip(M1_img * np.flip(M2_img, (0,1)),       0, 1)
    combined_270 = np.clip(M1_img * np.rot90(M2_img, -1),         0, 1)  # same as clockwise

    fig3, ax3 = plt.subplots(1, 4, figsize=(12, 3))
    titles = ["0° product", "90° product", "180° product", "270° product"]
    for i, img in enumerate([combined_0, combined_90, combined_180, combined_270]):
        ax3[i].imshow(img, cmap="gray"); ax3[i].set_title(titles[i]); ax3[i].axis("off")
    plt.tight_layout(); plt.show()


    save_img(M1_img, "plate1_gray.png")
    save_img(M2_img, "plate2_gray.png")
    print("Saved plate1_gray.png and plate2_gray.png (grayscale ink masks, 300 DPI)")