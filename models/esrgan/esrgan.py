import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import torchvision.models as models

# === Dataset ===
def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp'])

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform_lr=None, transform_hr=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted([f for f in os.listdir(lr_dir) if is_image_file(f)])
        self.hr_images = sorted([f for f in os.listdir(hr_dir) if is_image_file(f)])
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)
        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)

        return lr_img, hr_img

# === Transforms ===
transform_lr = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

transform_hr = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# === Paths ===
lr_dir = "./LR/LR"
hr_dir = "./HR/HR"
dataset = SRDataset(lr_dir, hr_dir, transform_lr, transform_hr)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# === RRDB Block (ESRGAN) ===
class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + 0.2 * x5

class RRDB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)

    def forward(self, x):
        return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))

# === Generator ===
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(64) for _ in range(23)])
        self.trunk_conv = nn.Conv2d(64, 64, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.upsample(fea)
        return torch.tanh(self.conv_last(fea))

# === Discriminator ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_channels, out_channels, stride):
            return [
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.net = nn.Sequential(
            *block(3, 64, 1),
            *block(64, 64, 2),
            *block(64, 128, 1),
            *block(128, 128, 2),
            *block(128, 256, 1),
            *block(256, 256, 2),
            *block(256, 512, 1),
            *block(512, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1)
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)

# === VGG for Perceptual Loss ===
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:36])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.features(x)

# === Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)
vgg = VGGFeatureExtractor().to(device)

l1 = nn.L1Loss()
bce = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

# === Pretraining ===
pretrain_epochs = 50
for epoch in range(pretrain_epochs):
    loop = tqdm(loader, desc=f"Pretrain Epoch [{epoch+1}/{pretrain_epochs}]")
    for lr, hr in loop:
        lr, hr = lr.to(device), hr.to(device)
        sr = generator(lr)
        loss = l1(sr, hr)

        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()

        loop.set_postfix(Pixel_Loss=loss.item())

# === Adversarial Training ===
adversarial_epochs = 300
for epoch in range(adversarial_epochs):
    loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{adversarial_epochs}]")
    for lr, hr in loop:
        lr, hr = lr.to(device), hr.to(device)
        sr = generator(lr)

        # --- Train Discriminator ---
        real_out = discriminator(hr)
        fake_out = discriminator(sr.detach())

        d_loss_real = bce(real_out - fake_out.mean(), torch.ones_like(real_out))
        d_loss_fake = bce(fake_out - real_out.mean(), torch.zeros_like(fake_out))
        d_loss = (d_loss_real + d_loss_fake) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # --- Train Generator ---
        # NOTE: Detach real_out here to prevent graph reuse
        with torch.no_grad():
            real_out_detached = discriminator(hr)

        fake_out = discriminator(sr)
        g_adv = bce(fake_out - real_out_detached.mean(), torch.ones_like(fake_out))
        g_percep = l1(vgg(sr), vgg(hr))
        g_pixel = l1(sr, hr)
        g_loss = 0.005 * g_adv + 0.01 * g_percep + g_pixel

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

# === Save Output ===
os.makedirs("./ESR_output_300", exist_ok=True) #"./SR_output"
generator.eval()
with torch.no_grad():
    for idx, (lr, hr) in enumerate(dataset):
        lr = lr.unsqueeze(0).to(device)
        sr = generator(lr)
        save_image(sr, f"./ESR_output_300/esr_{idx}.png")

torch.save(generator.state_dict(), "ESRGAN_generator_300.pth")
torch.save(discriminator.state_dict(), "ESRGAN_discriminator_300.pth")

# === PSNR/SSIM ===
sr_np = sr.squeeze().permute(1,2,0).cpu().numpy()
hr_np = hr.permute(1,2,0).cpu().numpy()
psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
ssim = structural_similarity(hr_np, sr_np, data_range=1.0, channel_axis=-1)
print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

def evaluate_sr_results(sr_folder, hr_folder):
    sr_filenames = sorted(os.listdir(sr_folder))
    hr_filenames = sorted(os.listdir(hr_folder))

    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    for sr_file, hr_file in zip(sr_filenames, hr_filenames):
        sr_image = Image.open(os.path.join(sr_folder, sr_file)).convert("RGB")
        hr_image = Image.open(os.path.join(hr_folder, hr_file)).convert("RGB")

        sr = np.array(sr_image)
        hr = np.array(hr_image)

        # Ensure same size for metric calculation
        if sr.shape != hr.shape:
            sr = Image.fromarray(sr).resize(hr_image.size, Image.BICUBIC)
            sr = np.array(sr)

        psnr_val = psnr(hr, sr, data_range=255)
        ssim_val = ssim(hr, sr, multichannel=True, data_range=255)

        psnr_total += psnr_val
        ssim_total += ssim_val
        count += 1

    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count

    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
evaluate_sr_results(sr_folder="./ESR_output_tuned", hr_folder="./HR/HR")

import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch

# Directories
sr_dir = "./ESR_output_tuned"
hr_dir = "./HR/HR"

# Load transform
transform = transforms.ToTensor()

# How many samples to display
num_samples = 5
sample_files = sorted(os.listdir(sr_dir))[:num_samples]

# Plotting
fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))

for i, sr_file in enumerate(sample_files):
    sr_path = os.path.join(sr_dir, sr_file)
    hr_path = os.path.join(hr_dir, sr_file.replace("esr_", "HR_"))  # assuming name pattern

    if os.path.exists(hr_path):
        sr_img = Image.open(sr_path).convert("RGB").resize((128, 128))
        hr_img = Image.open(hr_path).convert("RGB").resize((128, 128))

        axes[i, 0].imshow(sr_img)
        axes[i, 0].set_title("Super-Resolved (SR)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(hr_img)
        axes[i, 1].set_title("High-Resolution (HR)")
        axes[i, 1].axis("off")
    else:
        print(f"❌ Missing HR image for: {sr_file}")

plt.tight_layout()
plt.show()

generator.load_state_dict(torch.load("ESRGAN_generator_tuned.pth"))
discriminator.load_state_dict(torch.load("ESRGAN_discriminator_tuned.pth"))
