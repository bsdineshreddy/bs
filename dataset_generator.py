import rasterio
import numpy as np
import pandas as pd
from skimage.transform import resize
from pathlib import Path

# --- Project root ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Auto-find band files ---
green_path = list(ROOT.rglob("B03_GREEN.tif"))[0]
red_path   = list(ROOT.rglob("B04_RED.tif"))[0]
nir_path   = list(ROOT.rglob("B08_NIR.tif"))[0]

# --- Read bands ---
green = rasterio.open(green_path).read(1).astype(float)
red   = rasterio.open(red_path).read(1).astype(float)
nir   = rasterio.open(nir_path).read(1).astype(float)

# --- Resize ---
h = min(green.shape[0], red.shape[0], nir.shape[0])
w = min(green.shape[1], red.shape[1], nir.shape[1])

green = resize(green, (h, w), preserve_range=True)
red   = resize(red, (h, w), preserve_range=True)
nir   = resize(nir, (h, w), preserve_range=True)

# --- NDVI ---
ndvi = (nir - red) / (nir + red + 1e-10)

# --- Dataset ---
df = pd.DataFrame({
    "GREEN": green.flatten(),
    "RED": red.flatten(),
    "NIR": nir.flatten(),
    "NDVI": ndvi.flatten()
})

out_file = OUT_DIR / "ml_dataset.csv"
df.to_csv(out_file, index=False)

print("✅ DONE")
print(out_file)