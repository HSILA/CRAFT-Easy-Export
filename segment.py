from os import path
import utility as ut
from subprocess import call
import shutil
import argparse
import cv2
import os

parser = argparse.ArgumentParser(description='Easy Export Craft Utility')
parser.add_argument('--input', type=str, default='input', help='A Directory Containing Your Images.')
parser.add_argument('--output', type=str, default='output', help='Output Directory Containing Segmented Parts of Each Image.')
parser.add_argument('--pretrained', type=str, default='craft_mlt_25k.pth', help='Pretrained Model Name.')
parser.add_argument('--cuda',  type=str, default='True', help='Use Cuda for inference.')
parser.add_argument('--format', type=str, default='png', help='Output segments format.')
parser.add_argument('--original', type=str, default='False', help='Preserve original image with bounding boxes.')

args = parser.parse_args()

BASE_DIR = path.dirname(os.path.realpath(__file__))
INPUT_DIR = args.input
FULL_INPUT_DIR = path.join(BASE_DIR, INPUT_DIR)
OUTPUT_DIR = path.join(BASE_DIR, args.output)

ut.create_directory(args.output)

images_list = os.listdir(INPUT_DIR)
_ , file_extension = os.path.splitext(images_list[0])

if file_extension not in ['.jpg', '.jpeg']:
    new_input = path.join(BASE_DIR, 'input_jpg')
    ut.create_directory(new_input)
    ut.convert_to_jpg(FULL_INPUT_DIR, new_input)
    INPUT_DIR = new_input

call(f'python test.py --trained_model={args.pretrained} --test_folder={INPUT_DIR} --cuda={args.cuda}', shell=True)

CACHE_FULL_DIR = os.path.join(BASE_DIR, 'result')

for image in images_list:
    print(f'Processing {image_name}')
    
    image_name, _ = path.splitext(image)
    cv_img = cv2.imread(path.join(FULL_INPUT_DIR, image))
    out_dir = os.path.join(OUTPUT_DIR, image_name)

    ut.create_directory(out_dir)

    if args.original == 'True':
        shutil.copy(path.join(CACHE_FULL_DIR, f'res_{image_name}.jpg'), out_dir)

    coordinates = ut.coordinates_reader(path.join(CACHE_FULL_DIR, f'res_{image_name}.txt'))
    segment_counter = 1
    for coord in coordinates:
        segmented_image = ut.crop_image(cv_img, coord)
        segmented_image_path = path.join(out_dir, f'{image_name}-cropped-{segment_counter}.')
        if args.format == 'png':
            cv2.imwrite(segmented_image_path + 'png', segmented_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(segmented_image_path + args.format, segmented_image)

        segment_counter += 1

# Remove anything this code has created as cache
shutil.rmtree(path.join(BASE_DIR, 'result'))
shutil.rmtree(path.join(BASE_DIR, 'input_jpg'))