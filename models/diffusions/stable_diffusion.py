!pip install diffusers transformers accelerate safetensors

from diffusers import StableDiffusionUpscalePipeline

pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    revision="fp16",
    torch_dtype=torch.float16
).to("cuda")

# Load LR image
image = Image.open("/content/LR_670.png").convert("RGB").resize((128, 128))

# Run upscaling
prompt = "high-resolution astronomical image"
upscaled_image = pipe(prompt=prompt, image=image).images[0]
upscaled_image.save("upscaled_SR_image.jpg")

!pip install diffusers transformers accelerate safetensors
!pip install scikit-image

from diffusers import StableDiffusionUpscalePipeline
import torch
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

# --------------------------
#   Load Stable Diffusion Upscale Model
# --------------------------
pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    revision="fp16",
    torch_dtype=torch.float16
).to("cuda")

# --------------------------
#   Load LR & HR Images
# --------------------------
lr_image = Image.open("/content/LR_670.png").convert("RGB").resize((128, 128))
hr_image = Image.open("/content/HR_280.png").convert("RGB").resize((512, 512))  # Ground Truth HR

# --------------------------
#   Run Upscaling Model
# --------------------------
prompt = "high-resolution astronomical image"
sr_image = pipe(prompt=prompt, image=lr_image).images[0]
sr_image.save("/content/SR_670.png")

# --------------------------
#   Convert images to NumPy (for metrics)
# --------------------------
sr_np = np.array(sr_image)
hr_np = np.array(hr_image)

# Convert RGB to Y channel (luminance), more accurate for SR
sr_y = cv2.cvtColor(sr_np, cv2.COLOR_RGB2YCrCb)[:, :, 0]
hr_y = cv2.cvtColor(hr_np, cv2.COLOR_RGB2YCrCb)[:, :, 0]

# --------------------------
#   Compute PSNR & SSIM
# --------------------------
psnr_value = peak_signal_noise_ratio(hr_y, sr_y, data_range=255)
ssim_value = structural_similarity(hr_y, sr_y, data_range=255)

print("PSNR :", psnr_value)
print("SSIM :", ssim_value)

# Load LR image
image = Image.open("/content/LR_747.png").convert("RGB").resize((128, 128))

# Run upscaling
prompt = "high-resolution astronomical image"
upscaled_image = pipe(prompt=prompt, image=image).images[0]
upscaled_image.save("sd_2.jpg")

# Load LR image
image = Image.open("/content/LR_541.png").convert("RGB").resize((128, 128))

# Run upscaling
prompt = "high-resolution astronomical image"
upscaled_image = pipe(prompt=prompt, image=image).images[0]
upscaled_image.save("sd_3.jpg")

# Load LR image
image = Image.open("/content/LR_572.png").convert("RGB").resize((128, 128))

# Run upscaling
prompt = "high-resolution astronomical image"
upscaled_image = pipe(prompt=prompt, image=image).images[0]
upscaled_image.save("sd_4.jpg")

# Load LR image
image = Image.open("/content/LR_776.png").convert("RGB").resize((128, 128))

# Run upscaling
prompt = "high-resolution astronomical image"
upscaled_image = pipe(prompt=prompt, image=image).images[0]
upscaled_image.save("sd_5.jpg")

import requests

API_URL = ""
headers = {"Authorization": "Bearer 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"}

def query(image_path, prompt):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": prompt},
        files={"file": image_bytes}
    )
    return response.content

output = query("/content/LR_670.png", "high resolution photo of nebula")
with open("enhanced_SR_image.jpg", "wb") as f:
    f.write(output)

