import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import opening


image = np.load("wires/wires3.npy")
struct = np.ones((3,1))
process = opening(image,struct)

labeled_image = label(image)
labeled_process = label(process)

list_of_wires = np.unique(labeled_image)
for wire in range(1, len(list_of_wires)):
    labeled_wire_parts = label(opening(labeled_image == wire, struct))
    list_of_parts = np.unique(labeled_wire_parts)
    print(f"Wire {wire}: {len(list_of_parts)-1} parts")


print(f"Original {np.max(labeled_image)}")
print(f"Processed {np.max(labeled_process)}")

plt.imshow(labeled_process)
plt.show()
