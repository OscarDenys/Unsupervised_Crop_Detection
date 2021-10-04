from scipy.io import loadmat
import numpy as np
import os
from PIL import Image, ImageDraw
import sys
import random

sys.setrecursionlimit(1500)  # increase recursion limit to avoid stack overflow during line expansion

# different labels for squares in 8 by 8 grid:
#   vegetation detected in 8x8 resolution
#   blue and weed detection
#   blue but not tested
#   not blue and weed detection
#   crop line detection
#   blue and on crop line   } These two labels
#   crop line expansion     } are labeled as crop.


# FILE VARIABLES
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
original_img_name = 'RGB00417'
original_img_jpg_name = original_img_name + '.JPG'
original_img_path = os.path.join(THIS_FOLDER, original_img_jpg_name)
cropped_and_filtered_picture_name = original_img_name + '_cropped_and_filtered'
cropped_and_filtered_picture_jpg_name = cropped_and_filtered_picture_name + '.JPG'
cropped_and_filtered_img_path = os.path.join(THIS_FOLDER, cropped_and_filtered_picture_jpg_name)
crop_detection_name = original_img_name + '_crop_detection'
crop_detection_jpg_name = crop_detection_name + '.JPG'
crop_detection_img_path = os.path.join(THIS_FOLDER, crop_detection_jpg_name)
blue_matrix_name = original_img_name + '_blue_matrix.npy'
blue_matrix_path = os.path.join(THIS_FOLDER, blue_matrix_name)

# OTHER VARIABLES
# load line detection matrix
line_detect_dict = loadmat('line_detection_mask.mat')
line_detection_matrix = line_detect_dict['binary_mask']
# load  blue matrix
blue_matrix = np.load(blue_matrix_path, allow_pickle=True)
# swz = sliding window size
swz = 8  # 2656/8 = 332
number_of_rows = int(2656 / swz)
number_of_cols = number_of_rows
# colours
red = (255, 0, 0)
yellow = (255, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)
light_pink = (255, 204, 255)
orange = (255, 153, 51)
light_blue = (81, 237, 240)
light_green = (154, 226, 129)
curry = (205, 215, 23)


# FUNCTIONS


# Draws the red horizontal lines of the 8 by 8 grid in the 2656x2656 picture.
def draw_horizontal_red_lines(img_object):
    draw = ImageDraw.Draw(img_object)
    for i in range(number_of_rows):
        draw.line((0, swz * i, 2656, swz * i), fill=red)


# Draws a square in the 2656x2656 picture with a given color at a location specified in a 332x332  matrix.
def draw_square_at_row_col(row, col, img_object, color):
    # mindfuck warning row and col
    y = col * 8
    x = row * 8
    draw = ImageDraw.Draw(img_object)
    # vertical blue lines
    draw.line((x, y, x, y + swz), fill=color)  # left
    draw.line((x + swz, y, x + swz, y + swz), fill=color)  # right
    # horizontal line
    draw.line((x, y, x + swz, y), fill=color)  # upper
    draw.line((x, y + swz, x + swz, y + swz), fill=color)  # lower


# Draws a square with a given color in the 2656x2656 picture at the corresponding locations
# in a 332x332 matrix with a certain value.
def draw_matrix_at_value(img_object, matrix, value, color):
    for row in range(number_of_rows):
        for col in range(number_of_cols):
            if matrix[col, row] == value:
                draw_square_at_row_col(row, col, img_object, color)


# Draws a completely black square in the 2656x2656 picture at a location specified in a 332x332  matrix.
def obscure_at_row_col(row, col, img_object, color):
    # mindfuck warning row and col
    y = col * 8
    x = row * 8
    draw = ImageDraw.Draw(img_object)
    # vertical blue lines
    draw.line((x, y, x, y + swz), fill=color)  # left
    draw.line((x + swz, y, x + swz, y + swz), fill=color)  # right
    # horizontal line
    draw.line((x, y, x + swz, y), fill=color)  # upper
    draw.line((x, y + swz, x + swz, y + swz), fill=color)  # lower


# Draws a completely black square in the 2656x2656 picture at the corresponding locations
# in a 332x332 matrix with a certain value.
def obscure_matrix_at_value(img_object, matrix, value, color):
    for row in range(number_of_rows):
        for col in range(number_of_cols):
            if matrix[col, row] == value:
                draw_square_at_row_col(row, col, img_object, color)


# Auxiliary function for line expansion proces that returns the neighbouring positions to be checked in the
# following for loop.
def check_environment(row, col, blue_mat, line_mat, line_exp_mat, check_field=1, max_index=number_of_rows):
    positions = []
    # remember! : range (5,10) = 5,6,7,8,9
    row_range = range(row - check_field, row + check_field)
    if row < check_field:
        row_range = range(0, row + check_field)
    elif row > max_index - check_field:
        row_range = range(row - check_field, max_index)

    col_range = range(col - check_field, col + check_field)
    if col < check_field:
        col_range = range(0, col + check_field)
    elif col > max_index - check_field:
        col_range = range(col - check_field, max_index)

    for row in row_range:
        for col in col_range:
            if (blue_mat[row, col] == 1) and (line_mat[row, col] == 0) and (line_exp_mat[row, col] == 0):
                positions.append((row, col))
    random.shuffle(positions)
    return positions


# Recursively expand the crop rows.
def expand(current_pos, blue_mat, line_mat, line_exp_mat):
    positions = check_environment(current_pos[0], current_pos[1], blue_mat, line_mat, line_exp_mat)
    if len(positions) > 0:
        for new_pos in positions:
            line_exp_mat[new_pos[0], new_pos[1]] = 1
            expand(new_pos, blue_mat, line_mat, line_exp_mat)
    line_exp_mat[current_pos[0], current_pos[1]] = 1


# Make blue_on_line matrix indicating vegetation locations on crop lines. #############################################
sum_blue_and_line = line_detection_matrix + blue_matrix
blue_on_line_matrix = sum_blue_and_line
for r in range(number_of_rows):
    for c in range(number_of_cols):
        if sum_blue_and_line[r, c] == 2:
            blue_on_line_matrix[r, c] = 1
        else:
            blue_on_line_matrix[r, c] = 0



# Crop line expansion. #################################################################################################
line_expansion_mat = np.zeros((number_of_rows, number_of_cols), int)
# expanded_pos = []
for r in range(number_of_rows):
    for c in range(number_of_cols):
        if blue_on_line_matrix[r, c] == 1:
            for pos1 in check_environment(r, c, blue_matrix, line_detection_matrix, line_expansion_mat, 2):
                line_expansion_mat[pos1[0], pos1[1]] = 1
                for pos2 in check_environment(pos1[0], pos1[1], blue_matrix, line_detection_matrix, line_expansion_mat, 5):
                    line_expansion_mat[pos2[0], pos2[1]] = 1
                    for pos3 in check_environment(pos2[0], pos2[1], blue_matrix, line_detection_matrix, line_expansion_mat, 2):
                        expand(pos3, blue_matrix, line_detection_matrix, line_expansion_mat)
        if line_detection_matrix[r, c] == 1:
            for pos1 in check_environment(r, c, blue_matrix, line_detection_matrix, line_expansion_mat, 2):
                line_expansion_mat[pos1[0], pos1[1]] = 1
                for pos2 in check_environment(pos1[0], pos1[1], blue_matrix, line_detection_matrix, line_expansion_mat, 4):
                    line_expansion_mat[pos2[0], pos2[1]] = 1
                    for pos3 in check_environment(pos2[0], pos2[1], blue_matrix, line_detection_matrix, line_expansion_mat, 2):
                        expand(pos3, blue_matrix, line_detection_matrix, line_expansion_mat)
        if line_expansion_mat[r, c] == 1:
            for pos1 in check_environment(r, c, blue_matrix, line_detection_matrix, line_expansion_mat, 2):
                expand(pos1, blue_matrix, line_detection_matrix, line_expansion_mat)



# Make crop detection picture. #########################################################################################

crop_detection_matrix = blue_on_line_matrix + line_expansion_mat
crop_count = 0
blue_count = 0
for r in range(number_of_rows):
    for c in range(number_of_cols):
        if crop_detection_matrix[r, c] > 1:
            print('paniek, crop groter dan 2')
        elif crop_detection_matrix[r, c] == 1:
            crop_count += 1
        if blue_matrix[r,c] == 1:
            blue_count += 1
print(original_img_name, ' crop count: ', crop_count, ' blue count: ', blue_count)

img1 = Image.open(cropped_and_filtered_img_path).convert("RGB")
draw_matrix_at_value(img1, line_detection_matrix, 1, light_green)
draw_matrix_at_value(img1, blue_matrix, 1, blue)
draw_matrix_at_value(img1, crop_detection_matrix, 1, red)
draw_matrix_at_value(img1, blue_on_line_matrix, 1, light_pink)
img1.save(crop_detection_img_path, "JPEG")

# TODO fix output with black squares on crop locations
