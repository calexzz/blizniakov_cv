import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv, rgb2gray

def get_color(h):
    if h < 0.055:
        return "red"
    elif h < 0.11:
        return "orange"
    elif h < 0.22:
        return "yellow"
    elif h < 0.50:
        return "green"
    elif h < 0.67:
        return "blue"
    elif h < 0.83:
        return "purple"
    else:
        return "pink"

image = imread('./balls_and_rects.png')
img_hsv = rgb2hsv(image)

gray = rgb2gray(image)
binary = gray > 0.06
img_labeled = label(binary)
regions = regionprops(img_labeled)

color_counts = {"circles": {}, "rects": {}}

for region in regions:
    minr, minc, maxr, maxc = region.bbox
    mask = img_labeled[minr:maxr, minc:maxc] == region.label

    pixels_hsv = img_hsv[minr:maxr, minc:maxc][mask]
    mean_h = pixels_hsv[:, 0].mean()

    color = get_color(mean_h)
    shape = "circles" if region.extent < 0.9 else "rects"

    color_counts[shape][color] = color_counts[shape].get(color, 0) + 1

print("Всего фигур:", len(regions))
print("Круги по цветам:", color_counts["circles"])
print("Прямоугольники по цветам:", color_counts["rects"])