import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from model import VisionTransformer

# Load ImageNet labels
imagenet_labels = dict(enumerate(open("classes.txt")))

# Load model
model = torch.load("model.pth", weights_only=False)
model.eval()

# Hook function to extract attention maps
attn_maps = []
def get_attention_maps(module, input, output):
    attn_maps.append(output.detach())

# Register hook on the last attention layer
for name, module in model.named_modules():
    if "attn_drop" in name:  # Adjust for your ViT implementation
        module.register_forward_hook(get_attention_maps)

# Load and preprocess image
image_path = "panda.jpg"
img = Image.open(image_path).convert('RGB')
img = img.resize((384, 384))  # Ensure this matches model input size
img_np = np.array(img) / 255.0  # Normalize to [0, 1]

# Normalize with ImageNet statistics
img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

# Convert to tensor
inp = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

# Forward pass
with torch.no_grad():
    logits = model(inp)

# Get probabilities
probs = F.softmax(logits, dim=-1)
top_probs, top_ixs = probs[0].topk(10)

# Print top predictions
for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix].strip()
    print(f"{i}: {cls:<45} --- {prob:.4f}")

# Extract last attention map
attn = attn_maps[-1]  # Shape: (B, num_heads, num_tokens, num_tokens)
attn = attn.mean(dim=1).squeeze(0)  # Average over heads

# Compute grid size dynamically
num_tokens = attn.shape[-1] - 1  # Exclude CLS token
grid_size = int(num_tokens ** 0.5)  # Compute grid size dynamically

# Extract CLS token attention map
cls_attn = attn[0, 1:].reshape(grid_size, grid_size).cpu().numpy()

# Resize attention map to match original image size
cls_attn_resized = cv2.resize(cls_attn, (384, 384), interpolation=cv2.INTER_CUBIC)

# Normalize attention map for better visualization
cls_attn_resized = (cls_attn_resized - cls_attn_resized.min()) / (cls_attn_resized.max() - cls_attn_resized.min())

# Convert normalized image back to original range
img_np_orig = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
img_np_orig = np.clip(img_np_orig, 0, 1)  # Ensure values are within valid range

# Plot the attention overlay
plt.figure(figsize=(6, 6))
plt.imshow(img_np_orig)  # Show original image
plt.imshow(cls_attn_resized, cmap="jet", alpha=0.5)  # Overlay attention map
plt.axis("off")
plt.show()
