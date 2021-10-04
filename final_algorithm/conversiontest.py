import numpy as np
# IMPORTANT: PIL indexes images in col-row order, thus img is indexed as img[col,row]
from PIL import Image
import os
import cv2
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import time
from scipy.io import savemat

start_time = time.time()

# BOOLEAN SETTINGS
SET_ALL = False
save_filtered_img = False
save_generated_squares_array = False
save_weed_reconstruction_error_dictionary = False
save_crop_reconstruction_error_dictionary = False
save_weed_average_error_matrix = False
save_crop_average_error_matrix = False
save_test_counting_matrix = False
weed_draw_av_err_histogram = False
crop_draw_av_err_histogram = False
set_av_err_matr_to_minus_one_if_test_count_is_zero = False
save_separate_positive_matrix_equal_to_unaltered_av_err_matr = False
save_av_err_weed_minus_crop_matrix = False
draw_av_err_weed_minus_crop_histogram = True
if SET_ALL:
    save_filtered_img = SET_ALL
    save_generated_squares_array = SET_ALL
    save_weed_reconstruction_error_dictionary = SET_ALL
    save_crop_reconstruction_error_dictionary = SET_ALL
    save_weed_average_error_matrix = SET_ALL
    save_crop_average_error_matrix = SET_ALL
    save_weed_test_counting_matrix = SET_ALL
    save_crop_test_counting_matrix = SET_ALL
    weed_draw_av_err_histogram = SET_ALL
    crop_draw_av_err_histogram = SET_ALL
    set_av_err_matr_to_minus_one_if_test_count_is_zero = SET_ALL
    save_separate_positive_matrix_equal_to_unaltered_av_err_matr = SET_ALL
    save_av_err_weed_minus_crop_matrix = SET_ALL

# FILE VARIABLES
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
original_img_name = 'RGB00417'
original_img_jpg_name = original_img_name + '.JPG'
original_img_path = os.path.join(THIS_FOLDER, original_img_jpg_name)
filtered_picture_name = original_img_name + '_filtered'
filtered_picture_jpg_name = filtered_picture_name + '.JPG'
filtered_img_path = os.path.join(THIS_FOLDER, filtered_picture_jpg_name)
cropped_and_filtered_picture_name = original_img_name + '_cropped_and_filtered'
cropped_and_filtered_picture_jpg_name = cropped_and_filtered_picture_name + '.JPG'
cropped_and_filtered_img_path = os.path.join(THIS_FOLDER, cropped_and_filtered_picture_jpg_name)
# make sure that the following directory is empty before starting the algorithm
generated_squares_dir_path_4 = os.path.join(THIS_FOLDER, 'generated_squares_4')
generated_squares_array_npy_name = original_img_name + 'generated_squares_array_4.npy'
generated_squares_array_path_4 = os.path.join(THIS_FOLDER, generated_squares_array_npy_name)
filelist_generated_squares_4 = os.listdir(generated_squares_dir_path_4)
weed_saved_model_path = os.path.join(THIS_FOLDER, 'saved_model', 'weed_AE_38')
crop_saved_model_path = os.path.join(THIS_FOLDER, 'saved_model', 'crop_AE_38')
# OTHER VARIABLES
# range of HSV values for segmentation mask
lower_green = np.array([23, 50, 50])
upper_green = np.array([100, 255, 255])
# box defining the frame in order to crop the picture and retain the left hand side
box = (0, 0, 2656, 2656)  # left top right bottom
# swz = sliding window size
swz = 8
# 2656/8 = 332
number_of_rows = int(2656 / swz)
number_of_cols = number_of_rows
# number of pixels in a sliding window
num_pixels = swz * swz
# variable defining the amount of blue squares that are concatenated
field_size = int(swz / 2)
# latent dimension of autoencoders
latent_dimension = 38


# FUNCTIONS
# Generate blue matrix (332x332) containing positions of blue squares.
def make_blue_matrix(img_path):
    grey_scale_img = cv2.imread(img_path, 0)
    blue_mat = np.zeros((number_of_rows, number_of_cols), int)
    for row in range(number_of_rows):
        x = row * swz
        for col in range(number_of_cols):
            y = col * swz
            green_proportion = cv2.countNonZero(grey_scale_img[y:y + swz, x:x + swz]) / num_pixels  # col row order
            if green_proportion > 0.95:
                # !! think twice about this col row inversion and check the other 2 matrices
                blue_mat[col, row] = 1
    return blue_mat


# Auxiliary function in generate_squares for image names in the folder
def make_of_string_four_digits(number):
    number_string = str(number)
    length = len(number_string)
    if length == 4:
        return number_string
    if length == 3:
        return str(0) + number_string
    if length == 2:
        return str(0) + str(0) + number_string
    if length == 1:
        return str(0) + str(0) + str(0) + number_string


# Make a folder of relevant 32x32 images containing a minimum number of blue 8x8 squares.
def generate_squares(img_path, blue_mat, save_path, min_num_squares):
    # only for one picture
    save_path = os.path.join(save_path, 'example')
    img = Image.open(img_path).convert("RGB")
    for row in range(number_of_rows):
        for col in range(number_of_cols):
            if (row <= number_of_rows - field_size + 1) and (col <= number_of_cols - field_size + 1):
                numb_blue = cv2.countNonZero(blue_mat[row:row + field_size, col:col + field_size])
                if numb_blue >= min_num_squares:
                    left = col * swz
                    top = row * swz
                    right = left + (swz * field_size)
                    bottom = top + (swz * field_size)
                    box = (left, top, right, bottom)
                    square = img.crop(box)
                    # L = left, T = top
                    square_path = save_path + '_L' + make_of_string_four_digits(
                        left) + '_T' + make_of_string_four_digits(top) + '.JPG'
                    square.save(square_path, "JPEG")


# Updates the accounting matrices with the corresponding reconstruction error
# based on a pixel location in the original picture.
def update_accounting_matrices(left, top, recon_err, fault_accum_mat, test_counting_mat):
    start_col = int(left / swz)
    start_row = int(top / swz)
    for row_index in range(start_row - 1, start_row + field_size - 1):
        for column in range(start_col - 1, start_col + field_size - 1):
            fault_accum_mat[column, row_index] += recon_err
            test_counting_mat[column, row_index] += 1


# HSV BASED BACKGROUND SEGMENTATION  ###################################################################################
inimage = cv2.imread(original_img_path)
hsv_im = cv2.cvtColor(inimage, cv2.COLOR_BGR2HSV)  # better results if this is not the FULL conversion
# Threshold the HSV image to get only green colors
mask_green = cv2.inRange(hsv_im, lower_green, upper_green)
# Bitwise-AND mask and original image
filtered_img = cv2.bitwise_and(inimage, inimage, mask=mask_green)
# Convert back from HSV to RGB
filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_HSV2RGB_FULL)
if save_filtered_img:
    cv2.imwrite(filtered_img_path, filtered_img)
print("hsv filter loaded")

# REMOVE RIGHT SIDE OF PICTURE   #######################################################################################

# Convert from openCV2 to PIL (filtered_img is already in RGB thus no BGR2RGB conversion needed)
cropped_and_filtered_img = Image.fromarray(filtered_img)
# Crop the right part out of the image
cropped_and_filtered_img = cropped_and_filtered_img.crop(box)

cropped_and_filtered_img.save(cropped_and_filtered_img_path)
print("picture is cropped")

print('running time: ', "--- %s seconds ---" % (time.time() - start_time))
