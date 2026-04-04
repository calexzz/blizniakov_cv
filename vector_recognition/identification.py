import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops, perimeter
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0]+2, shape[1]+2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)

    return np.max(labeled)


def extractor(region):
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter / region.image.size
    holes = count_holes(region)
    vlines = (np.sum(region.image,0)==region.image.shape[0]).sum()
    hlines = (np.sum(region.image, 1) == region.image.shape[1]).sum()
    eccentricity = region.eccentricity
    aspect = region.image.shape[0] / region.image.shape[1]

    return np.array([region.area/region.image.size, cy, cx, perimeter,holes,vlines,hlines,eccentricity,aspect])

def classificator(region, templates):
    features = extractor(region)
    result = ""
    min_d = 10 ** 16
    for symbol, t in templates.items():
        d = ((t - features) ** 2).sum() ** 0.5
        if d < min_d:
            result = symbol
            min_d = d
    return result

template = imread('./alphabet-small.png')[:, :, :-1]
print(template.shape)
template = template.sum(2)
binary = template != 765

labeled = label(binary)
props = regionprops(labeled)

templates = {}

for region, symbol in zip(props, ["8", "0", "A", "B", "1", "W", "X", "*", "/", "-"]):

    templates[symbol] = extractor(region)

print(templates)
image = imread("./alphabet.png")[:, :, :-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
print(np.max(alabeled))
aprops = regionprops(alabeled)
res = {}

image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

print(count_holes(aprops[1]))

#plt.ion()
plt.figure(figsize=(5,7))
for region in aprops:
    symbol = classificator(region, templates)
    if symbol not in res:
        res[symbol] = 0
    res[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")
print(res)

plt.imshow(abinary)
plt.show()