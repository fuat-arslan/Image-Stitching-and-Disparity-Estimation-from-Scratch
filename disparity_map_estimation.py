# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import time

 # %%


def compute_disparity_pixel(x, y, left_img, right_img, window_size, max_disparity):
    pad = window_size // 2
    left_window = left_img[y - pad:y + pad + 1, x - pad:x + pad + 1]

    best_disparity = 65
    max_correlation = -1

    for d in range(1, min(max_disparity, x - pad + 1)):
        right_window = right_img[y - pad:y + pad + 1, x - d - pad:x - d + pad + 1]

        correlation = np.sum((left_window - np.mean(left_window)) * (right_window - np.mean(right_window)))
        correlation /= (np.std(left_window) * np.std(right_window) * window_size * window_size)
        
        # correlation = np.sum(left_window * right_window)

        if correlation > max_correlation:
            max_correlation = correlation
            best_disparity = d
            
    for d in range(1, min(max_disparity, left_img.shape[1] - pad - x)):
        right_window = right_img[y - pad:y + pad + 1, x + d - pad:x + d + pad + 1]
        
        correlation = np.sum((left_window - np.mean(left_window)) * (right_window - np.mean(right_window)))
        correlation /= (np.std(left_window) * np.std(right_window) * window_size * window_size)
        
        # correlation = np.sum(left_window * right_window)
                
        if correlation > max_correlation:
            max_correlation = correlation
            best_disparity = d

    return best_disparity

def compute_disparity_map(left_image_path, right_image_path, window_size=25, max_disparity=65):
    left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    height, width = left_img.shape
    disparity_map = np.zeros_like(left_img)
    
    pad = window_size // 2
    
    coords = [(x, y) for y in range(pad, height - pad) for x in range(pad, width - pad)]
    
    func = partial(compute_disparity_pixel, left_img=left_img, right_img=right_img, 
                   window_size=window_size, max_disparity=max_disparity)
    
    max_cpu_cores = cpu_count()
    with Pool(processes=max_cpu_cores) as pool:
        results = list(tqdm(pool.starmap(func, coords), total=len(coords)))

    idx = 0
    for y in range(pad, height - pad):
        for x in range(pad, width - pad):
            disparity_map[y, x] = results[idx]
            idx += 1
    
    return disparity_map

#%%
# example = str(input("Enter (cloth or plastic): ").strip().lower())

# if example != "cloth" and example != "plastic":
#     raise ValueError(f"example must be 'cloth' or 'plastic', not {example}")

# left_image_path = f'cv_proje/{example}/left.png'
# right_image_path = f'cv_proje/{example}/right.png'
# print("estimation started")
# t0 = time.time()
# disparity_map_left = compute_disparity_map(left_image_path, right_image_path, window_size=25)
# print("left is done")
# disparity_map_right = compute_disparity_map(right_image_path, left_image_path, window_size=25)
# print("right is done")
# print("estimation ended: ", time.time()-t0)

# cv2.imwrite(f"cv_proje/{example}/disp_left_out.png", disparity_map_left)
# cv2.imwrite(f"cv_proje/{example}/disp_right_out.png", disparity_map_right)

# disparity_left_gt = cv2.imread(f"cv_proje/{example}/disp_left.png")
# disparity_right_gt = cv2.imread(f"cv_proje/{example}/disp_right.png")

# # Create a figure and plot the images in a 2x2 grid
# fig, axs = plt.subplots(2, 2, figsize=(30, 30))

# # Plot ground truth disparities in the upper row
# axs[0, 0].imshow(cv2.cvtColor(disparity_left_gt, cv2.COLOR_BGR2RGB))
# axs[0, 0].set_title('Ground Truth Left Disparity')
# axs[0, 0].axis('off')

# axs[0, 1].imshow(cv2.cvtColor(disparity_right_gt, cv2.COLOR_BGR2RGB))
# axs[0, 1].set_title('Ground Truth Right Disparity')
# axs[0, 1].axis('off')

# # Plot predicted disparities in the lower row
# axs[1, 0].imshow(cv2.cvtColor(disparity_map_left, cv2.COLOR_BGR2RGB))
# axs[1, 0].set_title('Predicted Left Disparity')
# axs[1, 0].axis('off')

# axs[1, 1].imshow(cv2.cvtColor(disparity_map_right, cv2.COLOR_BGR2RGB))
# axs[1, 1].set_title('Predicted Right Disparity')
# axs[1, 1].axis('off')

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Save the resulting image
# plt.savefig(f"cv_proje/{example}/disparity_comparison.png")
# %%
