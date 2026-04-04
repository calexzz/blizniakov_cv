import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def count_bays(region):
    ch = count_holes(region)
    shape = region.image.shape
    padded = np.zeros((shape[0] + 2, shape[1] + 2))
    padded[1:-1, 1:-1] = region.image
    inverted = np.logical_not(padded)
    bays_labeled = label(inverted)
    x = sum(1 for r in regionprops(bays_labeled) if r.area < 3)
    return np.max(bays_labeled) - ch - x

def extractor(region):
    img = region.image.astype(float)
    h, w = img.shape
    cy, cx = region.centroid_local
    cy /= h
    cx /= w
    holes = count_holes(region)
    bays = count_bays(region)
    vlines = (np.sum(region.image, 0) == h).sum() / w
    area_ratio = region.area / img.size
    aspect = h / max(w, 1)
    eccentricity = region.eccentricity
    row_fill = np.sum(img, axis=1) / w
    col_fill = np.sum(img, axis=0) / h
    h_sym = 1 - np.abs(col_fill - col_fill[::-1]).mean()
    v_sym = 1 - np.abs(row_fill - row_fill[::-1]).mean()
    top_fill = img[:h//2, :].mean()
    bot_fill = img[h//2:, :].mean()
    left_fill = img[:, :w//2].mean()
    right_fill = img[:, w//2:].mean()
    orientation = region.orientation
    return np.array([area_ratio, cy, cx, holes, bays, vlines, eccentricity, aspect,
                     h_sym, v_sym, top_fill, bot_fill, left_fill, right_fill, orientation])

def weighted_dist(a, b):
    diff = a - b
    weights = np.array([
        3,   # area_ratio
        1,   # cy
        1,   # cx
        20,  # holes
        8,   # bays
        20,  # vlines
        20,  # eccentricity
        8,   # aspect
        20,  # h_sym
        15,  # v_sym
        4,   # top_fill
        4,   # bot_fill
        3,   # left_fill
        3,   # right_fill
        0,   # orientation
    ])
    return (weights * diff**2).sum() ** 0.5


def classificator(region, templates):
    features = extractor(region)
    res = ""
    min_dist = 10 ** 16
    for key in templates:
        current_dist = weighted_dist(templates[key], features)
        if current_dist < min_dist:
            min_dist = current_dist
            res = key
    return res

template = imread('./alphabet-small.png')[:, :, :-1]
template = template.sum(2)
binary = template != 765
labeled = label(binary)
props = regionprops(labeled)

templates = {}
for region, symbol in zip(props, ["8", "0", "A", "B", "1", "W", "X", "*", "/", "-"]):
    templates[symbol] = extractor(region)

image = imread("./alphabet.png")[:, :, :-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
aprops = regionprops(alabeled)

res = {}
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

plt.figure(figsize=(5, 7))
for region in aprops:
    symbol = classificator(region, templates)
    res[symbol] = res.get(symbol, 0) + 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")

print("Частотный словарь:", res)