# README for Image Processing Project

## Project Overview
This project is focused on two primary tasks in image processing:

1. **Image Stitching**: Combining multiple overlapping images to produce a seamless panoramic image.
2. **Disparity Map Estimation**: Computing disparity maps from stereo image pairs, which can be used for understanding depth information in scenes.

The project is structured into two main sections, each corresponding to the tasks mentioned above.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.x
- OpenCV library
- NumPy library
- Matplotlib library (optional, for visualization purposes)
- tqdm library (for progress tracking)

## Directory Structure
The project is organized into several directories:
- `outputs_Fuat_Melih`: This directory will be created automatically if it doesn't exist. It stores the output of both image stitching and disparity map estimation.
  - `Q1`: Contains subdirectories for each set of images used in image stitching.
    - `1`, `2`, `3`: Each subdirectory corresponds to a different group of images used for stitching.
  - `Q2`: Contains subdirectories for disparity map estimation outputs.
    - `cloth`, `plastic`: Each folder corresponds to different materials used in the disparity estimation process.

## Usage Instructions

### Image Stitching (`Q1`)
This part of the project combines multiple images to create panoramas. The function `Q1` in the script takes the path of the directory containing the images to be stitched.

To perform image stitching:
1. Place your sets of images in the `data/data_image_stitching` directory.
2. Each set should be named as `imX_Y.ext` where `X` is the set number and `Y` is the image number within the set.
3. Run the `__main__()` function. The stitched panoramas will be saved in `outputs_Fuat_Melih/Q1`.

### Disparity Map Estimation (`Q2`)
This section computes disparity maps for stereo image pairs. The function `Q2` in the script takes the path of the directory containing the stereo image pairs.

To compute disparity maps:
1. Place your stereo image pairs in the `data/data_disparity_estimation` directory.
2. In each material subdirectory (e.g., `cloth`, `plastic`), place the left and right images named as `left.png` and `right.png`.
3. Run the `__main__()` function. The disparity maps will be saved in `outputs_Fuat_Melih/Q2` under the respective material subdirectory.

## Output Visualization
The output images and disparity maps are saved in the respective directories within `outputs_Fuat_Melih`. You can view them using any standard image viewer.

## Troubleshooting
- Ensure all required libraries are installed.
- Check the format and naming of your input images.
- Verify that the input directories contain the correct images before running the script.


