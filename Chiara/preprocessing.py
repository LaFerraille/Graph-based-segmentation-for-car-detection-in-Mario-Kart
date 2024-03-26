import argparse
import logging
import time
from random import random
from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from model import segment_graph, build_graph

def diff(img, x1, y1, x2, y2):
    _out = np.sum((img[x1, y1] - img[x2, y2]) ** 2)
    return np.sqrt(_out)


def threshold(size, const):
    return (const * 1.0 / size)


def generate_image(forest, width, height):
    random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
    colors = [random_color() for i in range(width*height)]

    img = Image.new('RGB', (width, height))
    im = img.load()
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            im[x, y] = colors[comp]

    return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)


def get_yellow_mask(image_array,lower_yellow = np.array([20, 150, 150]),upper_yellow = np.array([50, 255, 255])):
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    return yellow_mask


def get_segmented_image(sigma, neighbor, K, min_comp_size, image_file):
    if neighbor != 4 and neighbor != 8:
        print('Invalid neighborhood chosen. The acceptable values are 4 or 8.')
        print('Segmenting with 4-neighborhood...')

    image_array = np.array(image_file)
    h,l,_size=image_array.shape
    size=l,h
    smooth = Image.fromarray(image_array).filter(ImageFilter.GaussianBlur(sigma))
    smooth = np.array(smooth).astype(int)

    graph_edges = build_graph(smooth, size[1], size[0], diff, neighbor == 8)

    forest = segment_graph(graph_edges, size[0] * size[1], K, min_comp_size, threshold)
    # im=generate_image(forest, size[1], size[0])
    # plt.imshow(im)
    # plt.show()
    return forest

def test_mask(input_file):
    image_file = Image.open(input_file)
    image_array = np.array(image_file)
    yellow_mask = get_yellow_mask(image_array)
    image_array[yellow_mask == 0] = [0, 0, 0]
    plt.imshow(image_array)
    plt.show()