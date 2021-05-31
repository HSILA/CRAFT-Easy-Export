from os import path
from typing import Counter
import new_utils
from subprocess import call
import argparse
import cv2
import os   

parser = argparse.ArgumentParser(description='Easy Export Craft Utility')
parser.add_argument('--input', type=str, default='input', help='A Directory Containing Your Images.')
parser.add_argument('--output', type=str, default='output', help='Output Directory Containing Segmented Parts of Each Image.')
parser.add_argument('--pretrained', type=str, default='craft_mlt_25k.pth', help='Pretrained Model Name.')

args = parser.parse_args()
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = args.input
FULL_INPUT_DIR = os.path.join(BASE_DIR, INPUT_DIR)
FULL_OUTPUT_DIR = os.path.join(BASE_DIR, args.output)

try:
    os.mkdir(args.output)
except:
    print('Output folder already exists, skip craeting it...')

images_list = os.listdir(INPUT_DIR)

call(f'python test.py --trained_model={args.pretrained} --test_folder={INPUT_DIR}', shell=True)

CACHE_FULL_DIR = os.path.join(BASE_DIR, 'result')

for image in images_list:
    cv_img = cv2.imread(os.path.join(FULL_INPUT_DIR, image))
    os.chdir(FULL_OUTPUT_DIR)
    image_name = image[:-4]
    os.mkdir(image_name)
    os.chdir(image_name)
    call(f'cp {CACHE_FULL_DIR}/res_{image} .', shell=True)
    coordinates = new_utils.coordinates_reader(f'{CACHE_FULL_DIR}/res_{image_name}.txt')
    number_counter = 1
    print(f'Processing {image}')
    for coord in coordinates:
        segmented_image = new_utils.crop_image(cv_img, coord)
        cv2.imwrite(f'cropped-{number_counter}.jpg', segmented_image)
        number_counter += 1

# Clear cache
call(f'rm -rf {BASE_DIR}/result', shell=True)