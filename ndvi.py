import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from pathlib import Path

# --- Project root ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Auto-find band files ---
red_path = list(ROOT.rglob("B04_RED.tif"))[0]
nir_path = list(ROOT.rglob("B08_NIR.tif"))[0]

# --- Read bands ---
red = rasterio.open(red_path).read(1).astype(float)
nir = rasterio.open(nir_path).read(1).astype(float)

# --- Resize to same shape ---
h = min(red.shape[0], nir.shape[0])
w = min(red.shape[1], nir.shape[1])

red = resize(red, (h, w), preserve_range=True)
nir = resize(nir, (h, w), preserve_range=True)

# --- NDVI ---
ndvi = (nir - red) / (nir + red + 1e-10)

plt.imshow(ndvi)
plt.colorbar(label="NDVI")
plt.title("NDVI Map")

out_file = OUT_DIR / "ndvi.png"
plt.savefig(out_file, dpi=300)
plt.show()

print("✅ DONE")
print(out_file)

