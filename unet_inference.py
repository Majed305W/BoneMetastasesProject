import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for post-processing
import pydicom
from torchvision import transforms
from PIL import Image

# ✅ Step 1: Define U-Net Model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[32, 64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottle_neck = DoubleConv(features[-1], features[-1] * 2)
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.max_pool(x)

        x = self.bottle_neck(x)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections.pop()
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.out_conv(x)

# ✅ Step 2: Load Trained U-Net Model
model = Unet(in_channels=1, out_channels=1)
model_path = './Model_50.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✅ U-Net model loaded successfully!")

# ✅ Step 3: Load and Preprocess DICOM Image
dicom_path = './downloaded_image.dcm'
if not os.path.exists(dicom_path):
    raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

ds = pydicom.dcmread(dicom_path)
image_array = ds.pixel_array.astype(np.float32)

# Normalize to [0, 1]
image_array -= np.min(image_array)
image_array /= np.max(image_array)

# Convert to PIL and resize to match model input
image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
image_pil = image_pil.resize((256, 256))

# Transform to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
image = transform(image_pil).unsqueeze(0).to(device)

# ✅ Step 4: Run Inference
with torch.no_grad():
    output = model(image)

# ✅ Step 5: Post-process Output
output = torch.sigmoid(output).squeeze().cpu().numpy()
print(f"Prediction min: {output.min()}, max: {output.max()}")

# Apply threshold
mask = (output > 0.2).astype(np.uint8)

# Clean mask with median filter
mask_cleaned = cv2.medianBlur(mask.astype(np.uint8), 5)

# ✅ Step 6: Save Segmentation Result with Fallback Check
os.makedirs('./results', exist_ok=True)
result_path = './results/segmentation_result.png'

try:
    plt.imshow(mask_cleaned, cmap='gray')
    plt.title("Segmentation Result (Bone & Tumors)")
    plt.axis('off')
    plt.savefig(result_path)
    print(f"✅ Segmentation result saved: {result_path}")
except Exception as e:
    print("❌ Failed to save segmentation result:", e)
