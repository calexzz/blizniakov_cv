import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

image = np.load('stars.npy')
labeled = label(image)

plus = 0
cross = 0

for r in regionprops(labeled):
    y1, x1, y2, x2 = r.bbox
    obj = image[y1:y2, x1:x2]

    if obj.shape == (5,5):
        if np.trace(obj) == 5:
            cross+=1
        elif obj[2,:].sum()==5:
            plus+=1


print(f"Plus: {plus}")
print(f"Cross: {cross}")

# plt.imshow(image)
# plt.show()