import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import data, color, transform
from PIL import Image

# =========================
# 0. 设备设置
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Running on:", device)

# =========================
# 1. 参数设置
# =========================
Nx, Ny = 256, 256          # 图像尺寸
num_iters = 10000
beta = 0.8  ##ATTENTION
alpha = 1e-2 ##ATTENTION
patch = 45  # 探针尺寸
step = 18 # 探针步长


# =========================
# 2. 加载 USAF 图片
# =========================


# 读灰度图
img = Image.open("USAF.jpg").convert("L")
usaf = np.array(img, dtype=float) / 255.0   # 归一化到 [0,1]

# resize 到 Nx, Ny
from skimage.transform import resize
usaf = resize(usaf, (Nx, Ny), anti_aliasing=True)

# 转成 Torch 张量
obj_amp = torch.tensor(usaf, dtype=torch.float32, device="mps")

# 构造相位（把灰度映射到 [-π/2, π/2]）
obj_phase = (obj_amp - obj_amp.min()) / (obj_amp.max() - obj_amp.min())
obj_phase = (obj_phase - 0.5) * torch.pi

# 构造复数透射函数
obj_true = obj_amp * torch.exp(1j * obj_phase)

# =========================
# 3. 探针 (高斯)
# =========================
x = torch.linspace(-1, 1, Nx, device=device)
y = torch.linspace(-1, 1, Ny, device=device)
X, Y = torch.meshgrid(x, y, indexing="ij")
probe = torch.exp(-((X)**2 + (Y)**2) / 0.05)

# =========================
# 4. 生成测量数据 (衍射强度)
# =========================
positions = [(i, j) for i in range(0, Nx - patch, step)
                     for j in range(0, Ny - patch, step)]
measured = []
for (px, py) in positions:
    obj_patch = obj_true[px:px+patch, py:py+patch]
    probe_patch = probe[px:px+patch, py:py+patch]
    exit_wave = obj_patch * probe_patch
    diff_wave = torch.fft.fftshift(torch.fft.fft2(exit_wave))
    measured.append(torch.abs(diff_wave))
measured = [m.to(device) for m in measured]

# =========================
# 5. 初始化对象猜测
# =========================
obj_guess = torch.ones_like(obj_true, dtype=torch.complex64, device=device)

# =========================
# 6. PI 迭代
# =========================
sse_list = []
for it in range(1, num_iters+1):
    total_err = 0.0
    count = 0
    for idx, (px, py) in enumerate(positions):
        pg = obj_guess[px:px+patch, py:py+patch] * probe[px:px+patch, py:py+patch]
        Wg = torch.fft.fftshift(torch.fft.fft2(pg))
        phase = torch.angle(Wg)
        Wc = measured[idx] * torch.exp(1j * phase)
        ce = torch.fft.ifft2(torch.fft.ifftshift(Wc))
        prob = probe[px:px+patch, py:py+patch]
        denom = (torch.abs(prob)**2 + alpha)
        upd = beta * (ce - pg) * torch.conj(prob) / denom
        obj_guess[px:px+patch, py:py+patch] += upd

        err = torch.sum((torch.abs(measured[idx])**2 - torch.abs(Wg)**2)**2).item()
        total_err += err
        count += Wg.numel()
    sse = total_err / count
    sse_list.append(sse)

    if it % 100 == 0:
        print(f"Iteration {it}/{num_iters}, SSE={sse:.3e}")

print("Finished.")

# =========================
# 7. 显示结果
# =========================
obj_true_np  = obj_true.cpu().numpy()
obj_guess_np = obj_guess.detach().cpu().numpy()

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.title("True Amplitude (USAF)")
plt.imshow(np.abs(obj_true_np), cmap="gray", origin="upper")
plt.colorbar()

plt.subplot(2,3,2)
plt.title("Reconstructed Amplitude")
plt.imshow(np.abs(obj_guess_np), cmap="gray", origin="upper")
plt.colorbar()

plt.subplot(2,3,4)
plt.title("True Phase")
plt.imshow(np.angle(obj_true_np), cmap="twilight", origin="upper")
plt.colorbar()

plt.subplot(2,3,5)
plt.title("Reconstructed Phase")
plt.imshow(np.angle(obj_guess_np), cmap="twilight", origin="upper")
plt.colorbar()

plt.subplot(1,3,3)
plt.title("SSE Convergence")
plt.semilogy(sse_list, "b-")
plt.xlabel("Iteration")
plt.ylabel("SSE (log scale)")

plt.tight_layout()
plt.savefig("ptycho_USAF_result.png", dpi=200)
plt.show()
