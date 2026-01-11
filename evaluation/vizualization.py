import cv2
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

image_paths = [
    '/content/LR_670.png', '/content/sd_1.jpg', '/content/HR_670.png',
    '/content/LR_747.png', '/content/sd_2.jpg', '/content/HR_47.png',
    '/content/LR_776.png', '/content/sd_5.jpg', '/content/HR_776.png'
]

num_images = len(image_paths)
cols = 3
rows = math.ceil(num_images / cols)

column_titles = ["Low Resolution", "ESRGAN", "High Resolution (Ground Truth)"]

fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=column_titles,
    horizontal_spacing=0.01,
    vertical_spacing=0.01
)

i = 0
for r in range(1, rows + 1):
    for c in range(1, cols + 1):
        if i < num_images:
            img = cv2.imread(image_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fig.add_trace(
                go.Image(z=img),
                row=r,
                col=c
            )
            i += 1

fig.update_layout(
    height=900,
    width=900,
    margin=dict(l=5, r=5, t=40, b=5),
    showlegend=False
)

fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

fig.show()

"""Stable Diffusion"""

import cv2
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

image_paths = [
    '/content/LR_926.png', '/content/esr_47.png', '/content/HR_926.png',
    '/content/LR_45.png', '/content/esr_310-2.png', '/content/HR_45.png',
    '/content/LR_675.png', '/content/esr_819.png', '/content/HR_675.png'
]

num_images = len(image_paths)
cols = 3
rows = math.ceil(num_images / cols)

column_titles = ["Low Resolution", "Stable Diffusion", "High Resolution (Ground Truth)"]

# ✅ KEY FIX: reduce spacing
fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=column_titles,
    horizontal_spacing=0.01,   # VERY SMALL GAP
    vertical_spacing=0.01      # VERY SMALL GAP
)

i = 0
for r in range(1, rows + 1):
    for c in range(1, cols + 1):
        if i < num_images:
            img = cv2.imread(image_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fig.add_trace(
                go.Image(z=img),
                row=r,
                col=c
            )
            i += 1

fig.update_layout(
    height=900,
    width=900,
    margin=dict(l=5, r=5, t=40, b=5),
    showlegend=False
)

fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

fig.show()

"""SRGAN"""

import cv2
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

image_paths = [
    '/content/LR_517.png', '/content/sr_output_tuned_216.png', '/content/HR_517.png',
    '/content/LR_813.png', '/content/sr_output_tuned_104.png', '/content/HR_813.png',
    '/content/LR_684.png', '/content/sr_output_tuned_251.png', '/content/HR_684.png'
]

num_images = len(image_paths)
cols = 3
rows = math.ceil(num_images / cols)

column_titles = ["Low Resolution", "SRGAN", "High Resolution (Ground Truth)"]

# ✅ KEY FIX: reduce spacing
fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=column_titles,
    horizontal_spacing=0.01,   # VERY SMALL GAP
    vertical_spacing=0.01      # VERY SMALL GAP
)

i = 0
for r in range(1, rows + 1):
    for c in range(1, cols + 1):
        if i < num_images:
            img = cv2.imread(image_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fig.add_trace(
                go.Image(z=img),
                row=r,
                col=c
            )
            i += 1

fig.update_layout(
    height=900,
    width=900,
    margin=dict(l=5, r=5, t=40, b=5),
    showlegend=False
)

fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

fig.show()

"""Restormer"""

print("PSNR:31.42   SSIM:0.7382")

image_paths = [
    '/content/rest.JPG',
    '/content/restormer.JPG',
]

for path in image_paths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

"""TransformerSR"""

print("PSNR:29.95 dB SSIM:0.7241")

image_paths = [
    '/content/tr.JPG',
    '/content/transformersr.JPG',
    '/content/trsr.JPG',
]

for path in image_paths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(
    header=dict(values=["Model","PSNR","SSIM"],
                fill_color='lightblue',
                align='center'),
    cells=dict(values=[
        ["ESRGAN", "SRGAN","Stable Diffusion","TransformerSR","Restormer"],
        [52.98,34.43,36.90,29.95,31.42],
        [0.8556,0.7437,0.79117,0.7241,0.7382]
    ],
    #fill_color='white',
    align='center'))
])

fig.show()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# ---------------- TABLE OF MODELS & METRICS ---------------- #

data = {
    "Model": ["SRGAN", "ESRGAN", "Stable Diffusion", "TransformerSR", "Restormer"],
    "PSNR (dB)": [34.43, 52.98, 36.90, 29.95, 31.42],
    "SSIM": [0.7437, 0.8556, 0.79117, 0.7241, 0.7382]
}

df = pd.DataFrame(data)
print(df)   # prints clean table in console