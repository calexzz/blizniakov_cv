import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label

def get_centroids(frame):
    labeled = label(frame)
    centroids = []
    for point in range(1, labeled.max()+1):
        coords = np.argwhere(labeled == point)
        cy = coords[:, 0].mean()
        cx = coords[:, 1].mean()
        centroids.append([cx, cy])
    return centroids

def find_nearest(point, candidates, used):
    px, py = point
    best_j = None
    best_dist = 100000

    for j, (cx, cy) in enumerate(candidates):
        if j in used:
            continue
        dist = (px - cx) ** 2 + (py - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_j = j

    return best_j

start_points = np.load('out/h_0.npy')
prev = get_centroids(start_points)
k_obj = 3

all_coords = np.zeros((100, k_obj, 2))
all_coords[0] = prev

for i in range(1, 100):
    d = np.load(f'out/h_{i}.npy')
    curr = get_centroids(d)

    used = set()
    for k in range(k_obj):
        near_j = find_nearest(prev[k], curr, used)
        used.add(near_j)
        all_coords[i,k] = curr[near_j]

    prev = all_coords[i]

for k in range(k_obj):
    xs = all_coords[:, k, 0]
    ys = all_coords[:, k, 1]
    plt.plot(xs, ys, marker='o', markersize=3, label=f'Объект {k+1}')

plt.gca().invert_yaxis()
plt.legend()
plt.show()