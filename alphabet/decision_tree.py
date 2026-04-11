import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops, perimeter
from skimage.io import imread
from pathlib import Path

def classificator(region):
    holes = count_holes(region)
    if holes == 2: #B, 8
        left_col = region.image[:, 0].sum() / region.image.shape[0]
        if left_col > 0.85:
            return "B"
        else:
            return "8"
    elif holes == 1: #A, 0, P, D
        img = region.image.astype(float)
        h, w = img.shape
        top_mass = img[:h // 2, :].sum() / img.sum()
        left_col = region.image[:, 0].sum() / h

        if left_col > 0.85:
            print(f"top_mass={top_mass:.3f}, left_col={left_col:.3f}, h={h}, w={w}")
            if top_mass > 0.53:
                return "P"
            else:
                return "D"
        else:
            if top_mass < 0.46:
                return "A"
            else:
                return "0"
    else: #1, W, X, *, -, /
        img = region.image.astype(float)
        h, w = img.shape
        aspect = np.min(region.image.shape) / np.max(region.image.shape)
        vlines = (np.sum(region.image, 0) == region.image.shape[0]).sum() / w
        # hlines = (np.sum(region.image, 1) == region.image.shape[1]).sum() / h
        fill = img.sum() / (h * w)

        if fill > 0.99:
            return "-"

        labeled = label(np.logical_not(region.image))
        bays = sum(1 for r in regionprops(labeled) if r.area > 3)

        if aspect > 0.85 and fill > 0.57 and vlines < 0.2:
            return "*"
        if bays >= 5: return "W"
        if bays == 4: return "X"

        if vlines > 0.2:  return "1"
        if fill > 0.55:
            return "*"
        else:
            return "/"
    return "?"

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0]+2, shape[1]+2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)

    return np.max(labeled) - 1

save_path = Path(__file__).parent

image = imread("./symbols.png")[:, :, :-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
aprops = regionprops(alabeled)

res = {}

image_path = save_path / "out_tree"
image_path.mkdir(exist_ok=True)

plt.figure(figsize=(5,7))
for region in aprops:
    symbol = classificator(region)
    if symbol not in res:
        res[symbol] = 0
    res[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")
print(res)
print(f"{1.0 - res.get('?',0) / len(aprops)}")