#%%
import cv2
import array
import random
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

#%%
def crop_image(image):
    mask = (image > 0).all(axis=2)  # Non-zero pixels
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped_image = image[y0:y1, x0:x1]
    return cropped_image

#%%
class FeatureMatcher:
    def __init__(self, descriptors1, descriptors2, method='linear'):
        self.descriptors1 = descriptors1
        self.descriptors2 = descriptors2
        self.method = method.lower()

        if self.method == 'linear':
            self.matcher = None
        else:
            raise ValueError("Invalid matching method. Choose 'kdtree', 'hashing', or 'linear'.")

    def nndr_match(self, threshold=0.7):
        matches = []

        for i in range(len(self.descriptors1)):
            desc1 = self.descriptors1[i]
            if self.method == 'linear':
                distances = np.linalg.norm(desc1 - self.descriptors2, axis=1)**2
                sorted = np.argsort(distances)
                best_match_index = sorted[0]
                second_best_match_index = sorted[1]
                nn_ratio = distances[best_match_index] / distances[second_best_match_index]
                if nn_ratio < threshold:
                    matches.append((i, best_match_index))
                continue

        return matches

#%%

class HomographyRANSAC:
    def __init__(self, num_iterations, inlier_threshold):
        self.num_iterations = num_iterations
        self.inlier_threshold = inlier_threshold

    def compute_homography(self, src_pts, dst_pts):
        A = []
        for src, dst in zip(src_pts, dst_pts):
            x, y = src
            u, v = dst
            # A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
            # A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
            A.append([-u, -v, -1, 0, 0, 0, x*u, x*v, x])
            A.append([0, 0, 0, -u, -v, -1, y*u, y*v, y])

        A = np.asarray(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1, :].reshape(3, 3)
        return H / H[2, 2]

    def apply_homography(self, H, points):
        points_homogeneous = np.column_stack((points, np.ones(len(points))))
        transformed_pts = np.dot(H, points_homogeneous.T).T
        transformed_pts /= transformed_pts[:, 2][:, np.newaxis]
        return transformed_pts[:, :2]

    def find_homography(self, kp1, kp2, matches):
        best_homography = None
        best_inliers = []

        all_src_pts = np.float32([kp1[idx1].pt for idx1, _ in matches])
        all_dst_pts = np.float32([kp2[idx2].pt for _, idx2 in matches])
        for _ in range(self.num_iterations):
            random_sample = random.sample(matches, 4)
            # random_sample = matches[:4]
            src_pts = np.float32([kp1[idx1].pt for idx1, _ in random_sample])
            dst_pts = np.float32([kp2[idx2].pt for _, idx2 in random_sample])

            candidate_homography = self.compute_homography(src_pts, dst_pts)

            
            transformed_pts = self.apply_homography(candidate_homography, all_dst_pts)
            distances = np.linalg.norm(transformed_pts - all_src_pts, axis=1)
            inliers = np.where(distances < self.inlier_threshold)[0]
            if len(inliers) > len(best_inliers):
                best_homography = candidate_homography
                best_inliers = inliers

        return best_homography, best_inliers

# Example usage
# homography_ransac = HomographyRANSAC(num_iterations=1000, inlier_threshold=5)
# H, inliers = homography_ransac.find_homography(kp1, kp2, matches)

#%%
def viz_matches(img1, img2, kp1, kp2, matches, best_inliers):
    img_concatenated = np.concatenate((img1, img2[:-1]), axis=1)

    plt.subplot(2,1,1)
    # Draw lines between matched points
    for match in matches:
        idx1, idx2 = match
        point1 = (int(kp1[idx1].pt[0]), int(kp1[idx1].pt[1]))
        point2 = (int(kp2[idx2].pt[0]) + img1.shape[1], int(kp2[idx2].pt[1]))

        # Draw line
        cv2.line(img_concatenated, point1, point2, color=(0, 255, 0), thickness=1)

    # Display the concatenated image with lines
    plt.title("Before RANSAC")
    plt.imshow(cv2.cvtColor(img_concatenated, cv2.COLOR_BGR2RGB))
    plt.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)

    ########################################################################################################################
    plt.subplots_adjust(hspace=0.5)

    img_concatenated = np.concatenate((img1, img2[:-1]), axis=1)

    plt.subplot(2,1,2)
    # Draw lines between inlier points after RANSAC
    for inlier_index in best_inliers:
        idx1, idx2 = matches[inlier_index]
        point1 = (int(kp1[idx1].pt[0]), int(kp1[idx1].pt[1]))
        point2 = (int(kp2[idx2].pt[0]) + img1.shape[1], int(kp2[idx2].pt[1]))

        # Draw line
        cv2.line(img_concatenated, point1, point2, color=(0, 255, 0), thickness=1)

    # Display the concatenated image with lines
    plt.title("After RANSAC")
    plt.imshow(cv2.cvtColor(img_concatenated, cv2.COLOR_BGR2RGB))
    plt.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)

    plt.show()

    print(f"Number of Feature Matches before RANSAC: {len(matches)}\nNumber of Feature Matches after  RANSAC: {len(best_inliers)}")

#%%

class ImageWarper:
    def __init__(self):
        pass

    def bilinear_interpolate(self, x, y):
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, self.image.shape[1] - 1)
        x1 = np.clip(x1, 0, self.image.shape[1] - 1)
        y0 = np.clip(y0, 0, self.image.shape[0] - 1)
        y1 = np.clip(y1, 0, self.image.shape[0] - 1)

        Ia = self.image[y0, x0]
        Ib = self.image[y1, x0]
        Ic = self.image[y0, x1]
        Id = self.image[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        return wa[..., np.newaxis] * Ia + wb[..., np.newaxis] * Ib + wc[..., np.newaxis] * Ic + wd[..., np.newaxis] * Id

    def warp_image(self, image, H, output_size, offset):
        self.image = image
        h, w = output_size
        ox, oy = offset
        warped_image = np.zeros((h, w, 3), dtype=self.image.dtype)

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack([x.flatten() - ox, y.flatten() - oy, np.ones_like(x.flatten())], axis=-1)

        H_inv = np.linalg.inv(H)
        transformed_coords = coords @ H_inv.T
        transformed_coords /= transformed_coords[:, 2, np.newaxis]

        x_transformed = transformed_coords[:, 0]
        y_transformed = transformed_coords[:, 1]

        for i in tqdm(range(h)):
            for j in range(w):
                xi, yi = x_transformed[i * w + j], y_transformed[i * w + j]
                if 0 <= xi < self.image.shape[1] - 1 and 0 <= yi < self.image.shape[0] - 1:
                    warped_image[i, j] = self.bilinear_interpolate(xi, yi)

        return warped_image

# Example usage
# image_warper = ImageWarper(image)
# warped_image = image_warper.warp_image(H, output_size, offset)

#%%
import numpy as np
cv = cv2
class ImageBlender:
    def __init__(self):
        pass

    def blend_with_gradient_alpha(self):
        alpha1 = (np.sum(self.img1, axis=2) > 0).astype(np.float32)
        alpha2 = (np.sum(self.img2, axis=2) > 0).astype(np.float32)

        rows, cols = alpha1.shape
        gradient = np.tile(np.linspace(0, 1, cols), (rows, 1))
        gradient[alpha1 == 0] = 0  # Zero out areas with no img1
        gradient[alpha2 == 0] = 1  # Set to 1 in areas with no img2

        blended = self.img1 * gradient[..., None] + self.img2 * (1 - gradient)[..., None]

        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def improved_blender(self, base_image, warped_image, top_pad=0):
        # Initial stitching setup
        stitched_image = warped_image.copy()
        stitched_image[top_pad:top_pad+crop_image(base_image).shape[0], 0:crop_image(base_image).shape[1]] = crop_image(base_image)

        # Create masks for both images
        mask_warped = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY) > 0
        mask_base = np.zeros_like(mask_warped)
        mask_base[top_pad:top_pad+crop_image(base_image).shape[0], 0:crop_image(base_image).shape[1]] = 1

        # Intersection of both images
        intersection = mask_base * mask_warped

        # Find the start and end of intersection along each row
        diff_intersection = np.diff(intersection.astype(int), axis=1)
        intersecting_rows = np.where(diff_intersection != 0)[1].reshape(-1, 2)

        # Calculate blending ratios
        blend_ratio_base = np.ones_like(intersection, dtype=float)
        blend_ratio_warped = np.ones_like(intersection, dtype=float)

        for idx, (start, end) in enumerate(intersecting_rows):
            start += 1
            range_length = end - start + 1
            
            blend_ratio_base[top_pad+idx, start:end+1] = (1 - np.linspace(0, 1, range_length))
            blend_ratio_warped[top_pad+idx, start:end+1] = np.linspace(0, 1, range_length)

        # Blending masks
        mask_blend_base = mask_base * blend_ratio_base
        mask_blend_warped = mask_warped * blend_ratio_warped
        # Alpha blending
        alpha_blended_image = (stitched_image * mask_blend_base[..., None]) + (warped_image * mask_blend_warped[..., None])
        # Return the blended image
        return alpha_blended_image
    
#%%
def find_output_size(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Corners of img1
    corners_img1 = np.array([[0, 0, 1], [w1, 0, 1], [0, h1, 1], [w1, h1, 1]])

    # Corners of img2
    corners_img2 = np.array([[0, 0, 1], [w2, 0, 1], [0, h2, 1], [w2, h2, 1]])
    corners_img2_transformed = (corners_img2 @ H.T)[:,:2]
    print(corners_img2_transformed)

    # Combine and find extremes
    all_corners = np.vstack((corners_img1[:, :2], corners_img2_transformed))
    print(all_corners)
    min_x, min_y = np.min(all_corners, axis=0)
    max_x, max_y = np.max(all_corners, axis=0)

    # Calculate size of the output image
    output_size = (int(np.ceil(max_y - min_y))*2, int(np.ceil(max_x - min_x))*2)
    offset = (-min_x, -min_y)

    return output_size, offset

#%%
def create_panorama(img1, img2, H, k = 1, top_pad=0):
    # output_size, offset = find_output_size(img1, img2, H)
    # ox, oy = offset

    # Ensure offset values are integers
    # ox, oy = int(ox), int(oy)

    # multiply by 2 for space for warped image
    output_size = (2**k*(img1.shape[1] + img2.shape[1]), 2**k*(img1.shape[0] + img2.shape[0]))
    
    wrapper = ImageWarper()
    img2_warped = wrapper.warp_image(img2, H, output_size, (0, 0))
    cv.imwrite("img2_warped.png", crop_image(img2_warped))
 
    blender = ImageBlender()
    blended_image = blender.improved_blender(img1, img2_warped, top_pad=top_pad)

    
    return crop_image(img2_warped), crop_image(blended_image)


#%%
def wrapper(img1_path_list, th=0.7, top_pad=0, output=None):
    blended_image = cv2.imread(img1_path_list[0])
    
    for i in range(len(img1_path_list)-1):
        img2_path = img1_path_list[i+1]
        img1 = blended_image if i == 0 else cv2.imread(os.path.join(output, 'blended_image.png'))
        img2 = cv2.imread(img2_path)

        top, bottom, left, right = top_pad, 0, 0, 0
        img1 = cv2.copyMakeBorder(img1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        print(img1.shape, img2.shape)
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        print('Feature Extraction Done')
        feature_matcher = FeatureMatcher(des1, des2, method='linear')
        matches = feature_matcher.nndr_match(th)
        print('Feature Matching Done')
        best_homography, _ = HomographyRANSAC(num_iterations=100000, inlier_threshold=5).find_homography(kp1, kp2, matches)
        print('RANSAC Done')
        print('Matching started')
        warped_img2, blended_image = create_panorama(img1, img2, best_homography, 1, top)
        print('Blending Done')
        if output is not None:
            cv2.imwrite(os.path.join(output, 'blended_image.png'), blended_image)
            cv2.imwrite(os.path.join(output, 'warped_image.png'), warped_img2)
    return blended_image

# #%%
# img1_path = '/auto/k2/farslan/CV_Project/CV_Homework/data/data_image_stitching/im1_1.png'
# img2_path = '/auto/k2/farslan/CV_Project/CV_Homework/data/data_image_stitching/im1_2.png'

# panoroma = wrapper([img1_path, img2_path], output='outputs/Q1_1')


# # %%
# img1_path = '/auto/k2/farslan/CV_Project/CV_Homework/data/data_image_stitching/im2_1.jpg'
# img2_path = '/auto/k2/farslan/CV_Project/CV_Homework/data/data_image_stitching/im2_2.jpg'
# img3_path = '/auto/k2/farslan/CV_Project/CV_Homework/data/data_image_stitching/im2_3.jpg'
# img4_path = '/auto/k2/farslan/CV_Project/CV_Homework/data/data_image_stitching/im2_4.jpg'

# panoroma = wrapper([img1_path, img2_path, img3_path, img4_path], output='outputs/Q1_2')

# # %%
# img1_path = '/auto/k2/farslan/CV_Project/CV_Homework/data/data_image_stitching/im3_0.jpg'
# img2_path = '/auto/k2/farslan/CV_Project/CV_Homework/data/data_image_stitching/im3_1.jpg'

# panoroma = wrapper([img1_path, img2_path], output='outputs/Q1_3')
