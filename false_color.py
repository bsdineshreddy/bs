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
green_path = list(ROOT.rglob("B03_GREEN.tif"))[0]
red_path   = list(ROOT.rglob("B04_RED.tif"))[0]
nir_path   = list(ROOT.rglob("B08_NIR.tif"))[0]

# --- Read bands ---
green = rasterio.open(green_path).read(1).astype(float)
red   = rasterio.open(red_path).read(1).astype(float)
nir   = rasterio.open(nir_path).read(1).astype(float)

# --- Resize to same shape ---
h = min(green.shape[0], red.shape[0], nir.shape[0])
w = min(green.shape[1], red.shape[1], nir.shape[1])

green = resize(green, (h, w), preserve_range=True)
red   = resize(red, (h, w), preserve_range=True)
nir   = resize(nir, (h, w), preserve_range=True)

# --- False color composite (R=NIR, G=RED, B=GREEN) ---
false_color = np.dstack((nir, red, green))
false_color /= false_color.max()

plt.imshow(false_color)
plt.title("False Color Composite")
plt.axis("off")

out_file = OUT_DIR / "false_color.png"
plt.savefig(out_file, dpi=300)
plt.show()

print("✅ DONE")
print(out_file)
