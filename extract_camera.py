import cv2
import numpy as np

save_id = 936
id = 491
frame = 45
color = np.load(f"data/collected_base/videos/{id:04d}.npy")
rgb = color[frame]

cv2.imwrite(f"outputs/output_{save_id}.png", rgb)