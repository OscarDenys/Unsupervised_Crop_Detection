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
original_img_name = 'RGB00468'
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
# green_proportion_threshold for selection in blue matrix
green_proportion_threshold = 0.80


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
            if green_proportion > green_proportion_threshold:
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

# LOAD BLUE MATRIX AND SQUARES ARRAY    ################################################################################

blue_matrix = make_blue_matrix(cropped_and_filtered_img_path)

blue_matrix.dump(original_img_name + '_blue_matrix.npy')
print('blue matrix is loaded')

generate_squares(cropped_and_filtered_img_path, blue_matrix, generated_squares_dir_path_4, 4)

for img in filelist_generated_squares_4:
    if img == '.DS_Store':
        filelist_generated_squares_4.remove('.DS_Store')
generated_squares_array = np.array(
    [np.array(Image.open(os.path.join(generated_squares_dir_path_4, fname))) for fname in filelist_generated_squares_4])
generated_squares_array = generated_squares_array.astype('float32') / 255.

if save_generated_squares_array:
    # make sure to adjust min_num_squares in name
    generated_squares_array.dump(original_img_name + '_generated_squares_array_4.npy')
print('generated_squares_array is loaded, shape: ', generated_squares_array.shape)

# **************************************************  WEED *************************************************************
print('WEED:')
#                 RECONSTRUCTION ERRORS ################################################################################

print('     start loading reconstruction errors')
weed_loaded_autoencoder = tf.keras.models.load_model(weed_saved_model_path)
# Check architecture of autoencoder
# print('     autoencoder summary:')
# weed_loaded_autoencoder.summary()

weed_encoded_imgs = np.zeros((len(generated_squares_array), latent_dimension), dtype=float)
for i in range(len(generated_squares_array)):
    if i == 0:
        continue
    elif i % 32 == 0:
        weed_encoded_imgs[i - 32:i - 1] = weed_loaded_autoencoder.encoder(generated_squares_array[i - 32:i - 1]).numpy()
    elif i == len(generated_squares_array):
        weed_encoded_imgs[i - i % 32:i - 1] = weed_loaded_autoencoder.encoder(
            generated_squares_array[i - i % 32:i - 1]).numpy()
print('     encoded images are loaded, shape:', weed_encoded_imgs.shape)

weed_decoded_imgs = np.zeros((len(generated_squares_array), 32, 32, 3), dtype=float)
for i in range(len(generated_squares_array)):
    if i == 0:
        continue
    elif i % 32 == 0:
        weed_decoded_imgs[i - 32:i - 1] = weed_loaded_autoencoder.decoder(weed_encoded_imgs[i - 32:i - 1]).numpy()
    elif i == len(generated_squares_array):
        weed_decoded_imgs[i - i % 32:i - 1] = weed_loaded_autoencoder.decoder(
            weed_encoded_imgs[i - i % 32:i - 1]).numpy()
print('     decoded images are loaded, shape: ', weed_decoded_imgs.shape)

j = 0
weed_reconstr_err_dict = {}
for square in filelist_generated_squares_4:
    if square == '.DS_Store':
        continue
    left = int(square[9:13])
    top = int(square[15:19])
    key = (left, top)
    mse = np.mean(np.power(generated_squares_array[j] - weed_decoded_imgs[j], 2))
    weed_reconstr_err_dict[key] = mse
    j += 1

if save_weed_reconstruction_error_dictionary:
    f = open(original_img_name + "_weed_reconstr_err_dict.pkl", "wb")
    pickle.dump(weed_reconstr_err_dict, f)
    f.close()
print('     reconstruction error dict is loaded')

#              AVERAGE ERROR MATRIX ####################################################################################

# blue_matrix_path = os.path.join(THIS_FOLDER, 'blue_matrix.npy')
# weed_reconstr_err_dict_path = os.path.join(THIS_FOLDER, 'weed_reconstr_err_dict.pkl')
# blue_matr = np.load(blue_matrix_path, allow_pickle=True)
# file_to_read = open(weed_reconstr_err_dict_path, "rb")
# weed_reconstr_err_dict = pickle.load(file_to_read)


# col major order -->  (col, row)
weed_fault_accum_matr = np.zeros((number_of_cols, number_of_rows))
test_counting_matr = np.zeros((number_of_cols, number_of_rows), int)
for key in weed_reconstr_err_dict.keys():
    L = int(key[0])
    T = int(key[1])
    update_accounting_matrices(L, T, weed_reconstr_err_dict[key], weed_fault_accum_matr, test_counting_matr)
weed_average_error_matrix = np.divide(weed_fault_accum_matr, test_counting_matr,
                                      where=test_counting_matr != 0)

if set_av_err_matr_to_minus_one_if_test_count_is_zero:
    # set average err matrix to -1 for elements where test count = 0
    for row in range(number_of_rows):
        for col in range(number_of_cols):
            if test_counting_matr[col, row] == 0:
                weed_average_error_matrix[col, row] = -1
    if save_separate_positive_matrix_equal_to_unaltered_av_err_matr:
        weed_positive_av_err_mat = weed_average_error_matrix.copy()
        for row in range(number_of_rows):
            for col in range(number_of_cols):
                if weed_average_error_matrix[col, row] == -1:
                    weed_positive_av_err_mat[col, row] = 0
        weed_positive_av_err_mat.dump(original_img_name + '_weed_positive_av_err_mat.npy')

if save_weed_average_error_matrix:
    weed_average_error_matrix.dump(original_img_name + '_weed_average_error_matrix.npy')
if save_test_counting_matrix:
    test_counting_matr.dump(original_img_name + '_test_counting_matrix.npy')
print('     average error matrix is loaded')

#               AVERAGE ERROR MATRIX HISTOGRAM #########################################################################

if weed_draw_av_err_histogram:
    av_err_array = weed_average_error_matrix.flatten()
    av_err_array = np.delete(av_err_array, np.where(av_err_array == -1))
    av_err_array = np.delete(av_err_array, np.where(av_err_array == 0))
    n1, bins1, patches1 = plt.hist(av_err_array, 100, facecolor='g')
    plt.xlabel("Err", fontweight="bold")
    plt.ylabel("# Occurences", fontweight="bold")
    plt.grid(True)
    plt.savefig(original_img_name + "_weed_average_error_distribution.png")
    plt.clf()

# **************************************************  CROP *************************************************************
print('CROP:')
#                 RECONSTRUCTION ERRORS ################################################################################

print('     start loading reconstruction errors')
crop_loaded_autoencoder = tf.keras.models.load_model(crop_saved_model_path)
# Check architecture of autoencoder
# print('     autoencoder summary:')
# crop_loaded_autoencoder.summary()

crop_encoded_imgs = np.zeros((len(generated_squares_array), latent_dimension), dtype=float)
for i in range(len(generated_squares_array)):
    if i == 0:
        continue
    elif i % 32 == 0:
        crop_encoded_imgs[i - 32:i - 1] = crop_loaded_autoencoder.encoder(generated_squares_array[i - 32:i - 1]).numpy()
    elif i == len(generated_squares_array):
        crop_encoded_imgs[i - i % 32:i - 1] = crop_loaded_autoencoder.encoder(
            generated_squares_array[i - i % 32:i - 1]).numpy()
print('     encoded images are loaded, shape:', crop_encoded_imgs.shape)

crop_decoded_imgs = np.zeros((len(generated_squares_array), 32, 32, 3), dtype=float)
for i in range(len(generated_squares_array)):
    if i == 0:
        continue
    elif i % 32 == 0:
        crop_decoded_imgs[i - 32:i - 1] = crop_loaded_autoencoder.decoder(crop_encoded_imgs[i - 32:i - 1]).numpy()
    elif i == len(generated_squares_array):
        crop_decoded_imgs[i - i % 32:i - 1] = crop_loaded_autoencoder.decoder(
            crop_encoded_imgs[i - i % 32:i - 1]).numpy()
print('     decoded images are loaded, shape: ', crop_decoded_imgs.shape)

j = 0
crop_reconstr_err_dict = {}
for square in filelist_generated_squares_4:
    if square == '.DS_Store':
        continue
    left = int(square[9:13])
    top = int(square[15:19])
    key = (left, top)
    mse = np.mean(np.power(generated_squares_array[j] - crop_decoded_imgs[j], 2))
    crop_reconstr_err_dict[key] = mse
    j += 1

if save_crop_reconstruction_error_dictionary:
    f = open(original_img_name + "_crop_reconstr_err_dict.pkl", "wb")
    pickle.dump(crop_reconstr_err_dict, f)
    f.close()
print('     reconstruction error dict is loaded')

#              AVERAGE ERROR MATRIX ####################################################################################

# blue_matrix_path = os.path.join(THIS_FOLDER, 'blue_matrix.npy')
# weed_reconstr_err_dict_path = os.path.join(THIS_FOLDER, 'weed_reconstr_err_dict.pkl')
# blue_matr = np.load(blue_matrix_path, allow_pickle=True)
# file_to_read = open(weed_reconstr_err_dict_path, "rb")
# weed_reconstr_err_dict = pickle.load(file_to_read)


# col major order -->  (col, row)
crop_fault_accum_matr = np.zeros((number_of_cols, number_of_rows))
crop_test_counting_matr = np.zeros((number_of_cols, number_of_rows), int)
for key in crop_reconstr_err_dict.keys():
    L = int(key[0])
    T = int(key[1])
    update_accounting_matrices(L, T, crop_reconstr_err_dict[key], crop_fault_accum_matr, crop_test_counting_matr)
crop_average_error_matrix = np.divide(crop_fault_accum_matr, crop_test_counting_matr,
                                      where=crop_test_counting_matr != 0)

if set_av_err_matr_to_minus_one_if_test_count_is_zero:
    # set average err matrix to -1 for elements where test count = 0
    for row in range(number_of_rows):
        for col in range(number_of_cols):
            if crop_test_counting_matr[col, row] == 0:
                crop_average_error_matrix[col, row] = -1
    if save_separate_positive_matrix_equal_to_unaltered_av_err_matr:
        crop_positive_av_err_mat = crop_average_error_matrix.copy()
        for row in range(number_of_rows):
            for col in range(number_of_cols):
                if crop_average_error_matrix[col, row] == -1:
                    crop_positive_av_err_mat[col, row] = 0
        crop_positive_av_err_mat.dump(original_img_name + '_crop_positive_av_err_mat.npy')

if save_crop_average_error_matrix:
    crop_average_error_matrix.dump(original_img_name + '_crop_average_error_matrix.npy')
print('     average error matrix is loaded')

#               AVERAGE ERROR MATRIX HISTOGRAM #########################################################################

if crop_draw_av_err_histogram:
    av_err_array = crop_average_error_matrix.flatten()
    av_err_array = np.delete(av_err_array, np.where(av_err_array == -1))
    av_err_array = np.delete(av_err_array, np.where(av_err_array == 0))
    n1, bins1, patches1 = plt.hist(av_err_array, 100, facecolor='g')
    plt.xlabel("Err", fontweight="bold")
    plt.ylabel("# Occurences", fontweight="bold")
    plt.grid(True)
    plt.savefig(original_img_name + "_crop_average_error_distribution.png")
    plt.clf()



# SAVE CROP HEATMAP MATRIX INTO MAT FILE FOR FOURIER TRANSFORMATION ####################################################

av_err_weed_minus_crop_matrix = weed_average_error_matrix - crop_average_error_matrix
if save_av_err_weed_minus_crop_matrix:
    av_err_weed_minus_crop_matrix.dump(original_img_name + '_average_error_weed_minus_crop_matrix.npy')
matrices_dict = {'av_err_diff_weed_minus_crop_matrix': av_err_weed_minus_crop_matrix,
                 'test_counting_matrix': test_counting_matr}
savemat(original_img_name + "_data.mat", matrices_dict)
print(original_img_name+' data saved to mat file')

# crop_thresholded_heatmap_matrix = av_err_weed_minus_crop_matrix.copy()
# threshold = np.mean(crop_thresholded_heatmap_matrix)
# for row in range(number_of_rows):
#    for col in range(number_of_cols):
#        if av_err_weed_minus_crop_matrix[col, row] < threshold:
#            crop_thresholded_heatmap_matrix[col, row] = 0
#        else:
#            crop_thresholded_heatmap_matrix[col, row] = 1

# if save_more_to_matlab_dict:
#    matrices_dict = {'crop_thresholded_heatmap_matrix': crop_thresholded_heatmap_matrix,
#                     'av_err_diff_weed_minus_crop_matrix': av_err_weed_minus_crop_matrix,
#                     'test_counting_matrix': test_counting_matr}
#    savemat(original_img_name + "_data.mat", matrices_dict)
#    print('crop_thresholded_heatmap_matrix and more saved to mat file')
# else:
#    matrices_dict = {'crop_thresholded_heatmap_matrix': crop_thresholded_heatmap_matrix}
#    savemat(original_img_name + "_data.mat", matrices_dict)
#    print('crop_thresholded_heatmap_matrix saved to mat file')

# WEED MINUS CROP AVERAGE ERROR MATRIX HISTOGRAM #########################################################################

if draw_av_err_weed_minus_crop_histogram:
    av_err_array = av_err_weed_minus_crop_matrix.flatten()
    av_err_array = np.delete(av_err_array, np.where(av_err_array == -1))
    av_err_array = np.delete(av_err_array, np.where(av_err_array == 0))
    n1, bins1, patches1 = plt.hist(av_err_array, 100, facecolor='g')
    plt.xlabel("Err", fontweight="bold")
    plt.ylabel("# Occurences", fontweight="bold")
    plt.grid(True)
    plt.savefig(original_img_name + "_average_error_weed_minus_crop_distribution.png")
    plt.clf()



print('running time: ', "--- %s seconds ---" % (time.time() - start_time))
print('running time per picture: ',
      "--- %s seconds ---" % ((time.time() - start_time) / generated_squares_array.shape[0]))
