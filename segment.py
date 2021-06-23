from os import path
import new_utils
from subprocess import call
import argparse
import cv2
import os   

parser = argparse.ArgumentParser(description='Easy Export Craft Utility')
parser.add_argument('--input', type=str, default='input', help='A Directory Containing Your Images.')
parser.add_argument('--output', type=str, default='output', help='Output Directory Containing Segmented Parts of Each Image.')
parser.add_argument('--pretrained', type=str, default='craft_mlt_25k.pth', help='Pretrained Model Name.')
parser.add_argument('--cuda',  default=True, help='Use Cuda for inference.')

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
_ , file_extension = os.path.splitext(images_list[0])

if file_extension != '.jpg':
    os.mkdir('new_input')
    new_utils.convert_to_jpg(os.path.join(BASE_DIR,INPUT_DIR), os.path.join(BASE_DIR,'new_input'))
    INPUT_DIR = 'new_input'
    FULL_INPUT_DIR = os.path.join(BASE_DIR, INPUT_DIR)
    os.chdir(BASE_DIR)

call(f'python test.py --trained_model={args.pretrained} --test_folder={INPUT_DIR} --cuda={args.cuda}', shell=True)

CACHE_FULL_DIR = os.path.join(BASE_DIR, 'result')

for image in images_list:
    image_name = image[:-4]
    cv_img = cv2.imread(os.path.join(FULL_INPUT_DIR, f'{image_name}.jpg'))
    os.chdir(FULL_OUTPUT_DIR)
    os.mkdir(image_name)
    os.chdir(image_name)
    call(f'cp {CACHE_FULL_DIR}/res_{image_name}.jpg .', shell=True)
    coordinates = new_utils.coordinates_reader(f'{CACHE_FULL_DIR}/res_{image_name}.txt')
    number_counter = 1
    print(f'Processing {image}')
    for coord in coordinates:
        segmented_image = new_utils.crop_image(cv_img, coord)
        cv2.imwrite(f'{image_name}-cropped-{number_counter}.jpg', segmented_image)
        number_counter += 1

# Remove anything this code has created as cache
call(f'rm -rf {BASE_DIR}/result', shell=True)
call(f'rm -rf {BASE_DIR}/new_input', shell=True)