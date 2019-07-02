from glob import glob
from PIL import Image
from random import choice
import pandas as pd
from tqdm import tqdm


def get_crops(img, x_from, y_from, x_to, y_to, size):
    width, height = x_to - x_from, y_to - y_from

    x_count = width // size + 1
    step_x = max(int(0.5 + (width - size) / x_count), 1)
    x_to = x_from + width - size + 1

    y_count = height // size + 1
    step_y = max(int(0.5 + (height - size) / y_count), 1)
    y_to = y_from + height - size + 1
#     print(y_count, y_from ,y_to, step_y)

    crops = []
    shifts = []
    for y in range(y_from, y_to, step_y):
        for x in range(x_from, x_to, step_x):
            crops.append(img.crop((x, y, x+size, y+size)))
            shifts.append((x, y))

    return crops, shifts


path = '/data/ice/images/2018-03-07_1336_right/025520.jpg'
img = Image.open(path)
crops, shifts = get_crops(img, 0, 0, 2448, 1208, 608)
