
import cv2
import os

def coordinates_reader(file_path):
    coordinates = []
    lines = []
    with open(file_path, 'r') as file:
        whole = file.read()
        lines = whole.split('\n')
    lines = lines[:-1]
    for line in lines:
        coordinates.append(non_zero_list([int(item) for item in line.split(',')]))
    return coordinates

def convert_to_jpg(source_folder, destination_folder):
    images_list = os.listdir(source_folder)
    os.chdir(source_folder)

    for image in images_list:
        img = cv2.imread(image)
        cv2.imwrite(os.path.join(destination_folder, image[:-3] + 'jpg'), img)

def crop_image(image, corners):
    xs = [corners[i] for i in range(8) if i % 2 == 0]
    ys = [corners[i] for i in range(8) if i % 2 != 0]
    top_left_x = min(xs)
    top_left_y = min(ys)
    bot_right_x = max(xs)
    bot_right_y = max(ys)
    return image[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]

def non_zero_list(the_list):
    list_size = len(the_list)
    for i in range(list_size):
        if the_list[i] < 0:
            the_list[i] = 1
    return the_list
