import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# MRI file name (exact file in your folder)
nii_path = "BraTS20_Training_369_flair.nii"

# Load MRI file
img = nib.load(nii_path)
data = img.get_fdata()

print("Image shape:", data.shape)

# Create folder for output
os.makedirs("slices", exist_ok=True)

# Save slices as PNG
for i in range(data.shape[2]):
    plt.imsave(f"slices/slice_{i}.png", data[:, :, i], cmap="gray")

print("DONE — PNG slices saved in 'slices' folder")