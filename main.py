import os
import cv2
import random
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import time


from image_stiching import wrapper
from disparity_map_estimation import compute_disparity_map


if os.path.exists("outputs_Fuat_Melih"):
    print("outputs_Fuat_Melih directory already exists")
else:
    print("Creating outputs_Fuat_Melih directory")
    os.mkdir("outputs_Fuat_Melih")
    os.mkdir("outputs_Fuat_Melih/Q1")
    os.mkdir("outputs_Fuat_Melih/Q1/1")
    os.mkdir("outputs_Fuat_Melih/Q1/2")
    os.mkdir("outputs_Fuat_Melih/Q1/3")
    os.mkdir("outputs_Fuat_Melih/Q2")
    os.mkdir("outputs_Fuat_Melih/Q2/cloth")
    os.mkdir("outputs_Fuat_Melih/Q2/plastic")

# Q1
def Q1(root_path_stitch):
    # Q1.1
    Q1_img1_path = os.path.join(root_path_stitch, 'im1_1.png')
    Q1_img2_path = os.path.join(root_path_stitch, 'im1_2.png')
    save_Directory = 'outputs_Fuat_Melih/Q1/1'
    print("Computing panorama for Q1.1")
    panoroma1 = wrapper([Q1_img1_path , Q1_img2_path], output=save_Directory)
    print("Done")

    # Q1.2
    Q1_img1_path = os.path.join(root_path_stitch, 'im2_1.jpg')
    Q1_img2_path = os.path.join(root_path_stitch, 'im2_2.jpg')
    Q1_img3_path = os.path.join(root_path_stitch, 'im2_3.jpg')
    Q1_img4_path = os.path.join(root_path_stitch, 'im2_4.jpg')
    save_Directory = 'outputs_Fuat_Melih/Q1/2'
    print("Computing panorama for Q1.2")
    panoroma2 = wrapper([Q1_img1_path , Q1_img2_path, Q1_img3_path, Q1_img4_path], output=save_Directory)
    print("Done")

    # Q1.3
    Q1_img1_path = os.path.join(root_path_stitch, 'im3_0.jpg')
    Q1_img2_path = os.path.join(root_path_stitch, 'im3_1.jpg')
    save_Directory = 'outputs_Fuat_Melih/Q1/3'
    print("Computing panorama for Q1.3")
    panoroma3 = wrapper([Q1_img1_path , Q1_img2_path], output=save_Directory)
    print("Done")

    return panoroma1, panoroma2, panoroma3

# Q2
def Q2(root_path_disparity):
    #cloth
    Q2_left_path = os.path.join(root_path_disparity, 'cloth/left.png')
    Q2_right_path = os.path.join(root_path_disparity, 'cloth/right.png')
    save_Directory = 'outputs_Fuat_Melih/Q2/cloth'
    print("Computing disparity map for cloth")
    disparity_map_left_cloth = compute_disparity_map(Q2_left_path, Q2_right_path, window_size=25)
    print("left is done")
    disparity_map_right_cloth = compute_disparity_map(Q2_right_path, Q2_left_path, window_size=25)
    print("right is done")
    print("Saving disparity map for cloth")
    cv2.imwrite(f"{save_Directory}/disp_left_out.png", disparity_map_left_cloth)
    cv2.imwrite(f"{save_Directory}/disp_right_out.png", disparity_map_right_cloth)
    print("Done")

    #plastic
    Q2_left_path = os.path.join(root_path_disparity, 'plastic/left.png')
    Q2_right_path = os.path.join(root_path_disparity, 'plastic/right.png')
    save_Directory = 'outputs_Fuat_Melih/Q2/plastic'
    print("Computing disparity map for plastic")
    disparity_map_left_plastic = compute_disparity_map(Q2_left_path, Q2_right_path, window_size=25)
    print("left is done")
    disparity_map_right_plastic = compute_disparity_map(Q2_right_path, Q2_left_path, window_size=25)
    print("right is done")
    print("Saving disparity map for plastic")
    cv2.imwrite(f"{save_Directory}/disp_left_out.png", disparity_map_left_plastic)
    cv2.imwrite(f"{save_Directory}/disp_right_out.png", disparity_map_right_plastic)
    print("Done")

    return disparity_map_left_plastic, disparity_map_right_plastic, disparity_map_left_cloth, disparity_map_right_cloth


def __main__():

    # Q1
    root_path_stitch = 'data/data_image_stitching'
    panoroma1, panoroma2, panoroma3 = Q1(root_path_stitch)
    # Q2
    root_path_disparity = 'data/data_disparity_estimation'
    disparity_map_left_plastic, disparity_map_right_plastic, disparity_map_left_cloth, disparity_map_right_cloth = Q2(root_path_disparity)

if __name__ == "__main__":
    __main__()





 



