import torch
from diffusers import AutoencoderKL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define the model repository from Hugging Face Model Hub
vae_model = "stabilityai/sd-vae-ft-ema"

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

# Load the pre-trained AutoencoderKL model
model = AutoencoderKL.from_pretrained(vae_model).to(device)

# Ensure the model is in evaluation mode
model.eval()

# Generate random latent vectors
latent_vectors = torch.randn(1, 4, 64, 64)  # Adjust dimensions as necessary

# Generate images from latent vectors
with torch.no_grad():
    output = model.decode(latent_vectors)

# # Move generated_images to CPU and detach from computation graph
generated_images = output.sample
generated_images=generated_images.cpu().detach()

# Convert tensor to numpy array
img_np = generated_images.squeeze(0).permute(1, 2, 0).numpy()

# Normalize the image data to the range [0, 1]
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

# Save the PIL image as a PNG file
img_pil.save("generated_image.png")

# Optionally, display the image
plt.imshow(img_pil)
plt.axis('off')
plt.show()