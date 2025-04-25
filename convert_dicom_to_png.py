# convert_dicom_to_png.py
import pydicom
import numpy as np
from PIL import Image

ds = pydicom.dcmread("downloaded_image.dcm")
image = ds.pixel_array.astype(np.float32)
image -= np.min(image)
image /= np.max(image)

image_pil = Image.fromarray((image * 255).astype(np.uint8))
image_pil = image_pil.resize((256, 256))
image_pil.save("results/original_image.png")
print("✅ DICOM converted to results/original_image.png")
