import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from skimage.io import imread, imsave
from skimage import img_as_float
from skimage.color import rgb2gray

# Paths
img_path = "normal/normal (2).png"
model_path = "DnCNN-keras/snapshot/save_DnCNN_sigma25_2025-08-02-21-12-53/model_50.h5"

# Load and normalize image (grayscale)
img = imread(img_path)
if img.ndim == 3:
    img = rgb2gray(img)
img = img_as_float(img).astype(np.float32)

# Prepare input
inp = img[np.newaxis, ..., np.newaxis]

# Load model without compiling first
model = load_model(model_path, compile=False)

# Compile the model with a standard loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# Apply model
try:
    denoised = model.predict(inp, verbose=0)
    denoised = np.clip(np.squeeze(denoised), 0, 1)
except Exception as e:
    print(f"Error during prediction: {e}")
    exit(1)

# Visualize or save
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(denoised, cmap='gray'); plt.title("DnCNN Denoised"); plt.axis('off')
plt.tight_layout()
plt.show()

imsave("denoised.png", (denoised * 255).astype(np.uint8))
