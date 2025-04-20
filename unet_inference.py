import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for post-processing
from torchvision import transforms
from PIL import Image
import os

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

        # Encoder (downsampling path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck & Decoder (upsampling path)
        self.bottle_neck = DoubleConv(features[-1], features[-1] * 2)
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

    def forward(self, x):
        skip_connections = []

        # Encoder (downsampling path)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.max_pool(x)

        x = self.bottle_neck(x)

        # Decoder (upsampling path)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections.pop()

            # Ensure the feature map sizes match
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.out_conv(x)


# ✅ Step 2: Load the Trained Model
model = Unet(in_channels=1, out_channels=1)

# ✅ Load the State Dictionary (Trained Weights)
model_path = './model/Model_50.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights
model.load_state_dict(torch.load(model_path, map_location=device))

# Move model to the correct device (CPU or GPU)
model.to(device)
model.eval()

print("✅ U-Net model loaded successfully!")


# ✅ Step 3: Load and Preprocess a Test Image
image_path = './test_images/example.png'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Test image not found: {image_path}")

# Define image transformations (🚀 Removed normalization)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load and preprocess the image
image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)

# ✅ Step 4: Visualize the Model Input (Debugging)
plt.imshow(image.cpu().squeeze().numpy(), cmap='gray')
plt.title("Model Input Image")
plt.axis('off')
# plt.show()  # Disabled to avoid popup during server run

# ✅ Step 5: Run Inference
with torch.no_grad():
    output = model(image)

# ✅ Step 6: Post-processing (Thresholding)
output = torch.sigmoid(output).squeeze().cpu().numpy()
print(f"Min: {output.min()}, Max: {output.max()}")

# 🚀 Lower threshold to capture both bone & tumor regions
mask = (output > 0.2).astype(np.uint8)

# ✅ Step 7: Post-Processing with OpenCV (Remove Small Artifacts)
mask_cleaned = cv2.medianBlur(mask.astype(np.uint8), 5)  # Apply median filter to smooth

# ✅ Step 8: Save and Display the Segmentation Result
result_path = './results/segmentation_result.png'
plt.imshow(mask_cleaned, cmap='gray')
plt.title("Segmentation Result (Bone & Tumors)")
plt.axis('off')
plt.savefig(result_path)
# plt.show()  # Disabled to avoid popup during server run

print(f"✅ Segmentation result saved: {result_path}")