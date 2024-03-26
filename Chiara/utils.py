import math
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
import cv2

def nice_plot(forest, original_image):
    comp_list=[]
    height,width = original_image.size
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            comp_list.append(comp)
    unique, counts = np.unique(comp_list, return_counts=True)
    dictio=dict(zip(unique, counts))
    # cle_minimum = min(dictio, key=dictio.get)

    original_image = original_image.convert("RGBA")
    g=256
    for y in range(height):
        for x in range(width):
            # if comp_list[y * width + x] == cle_minimum:
            if dictio[comp_list[y * width + x]] < 400 and y>30 and y<130:
                original_image.putpixel((y, x), (g, g, g))
    # plt.imshow(original_image)
    # plt.show()
    return original_image

def nice_plot_bis(forest, init_image, original_image,fac):
    comp_list=[]
    height,width = original_image.size
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            comp_list.append(comp)
    unique, counts = np.unique(comp_list, return_counts=True)
    dictio=dict(zip(unique, counts))
    cle_minimum = max(dictio, key=dictio.get)

    centre_x = width // 2
    centre_y = height *2 // 3
    r = int(min(width, height) * 0.2)

    mask = np.zeros((width,height))
    g=256
    for y in range(height):
        for x in range(width):
            if comp_list[y * width + x] != cle_minimum and dictio[comp_list[y * width + x]] < 400:
                if math.sqrt((x - centre_x) ** 2 + (y - centre_y) ** 2) <= r:
            # if dictio[comp_list[y * width + x]] < 400 and y>30 and y<130:
                    mask[x, y] = 1
    mask_big = cv2.resize(mask, (fac*height, fac*width), interpolation=cv2.INTER_NEAREST)
    new_image = np.array(init_image)
    new_image[mask_big == 1] = [g, g, g]
    new_image=PILImage.fromarray(new_image).convert("RGB")
    plt.imshow(new_image)
    plt.show()
    return new_image

def nice_plot_ter(forest, init_image, original_image,fac):
    from collections import defaultdict

    height,width = original_image.size

    components_points = defaultdict(list)
    components_centers = {}

    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)        
            components_points[comp].append((x, y))

    for comp, points in components_points.items():
        avg_x = sum(point[0] for point in points) / len(points)
        avg_y = sum(point[1] for point in points) / len(points)
        
        components_centers[comp] = (avg_x, avg_y)

    filtered_components = {comp: points for comp, points in components_points.items() if len(points) < 700}

    centre_x = 60 #width // 2
    centre_y = 80 #height *2 // 3

    distances = {comp: ((centre_x - center[0]) ** 2 + (centre_y - center[1]) ** 2) ** 0.5 for comp, center in components_centers.items() if comp in filtered_components}
    sorted_components = sorted(distances.items(), key=lambda x: x[1])

    # for comp, distance in sorted_components:
    #     print(f"Composante {comp}: Distance au point défini = {distance} et taille = {len(filtered_components[comp])}")
    if len(sorted_components) == 0:
        return init_image
    chosen = sorted_components[0][0]
    # print(chosen)
    mask = np.zeros((width,height))
    g=256
    for points in components_points[chosen]:
                    mask[points] = 1
    mask_big = cv2.resize(mask, (fac*height, fac*width), interpolation=cv2.INTER_NEAREST)
    new_image = np.array(init_image)
    new_image[mask_big == 1] = [g, g, g]
    new_image=PILImage.fromarray(new_image).convert("RGB")
    plt.imshow(new_image)
    plt.show()
    return new_image


def get_mask(forest, original_image):
    comp_list=[]
    height,width = original_image.size
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            comp_list.append(comp)

    unique, counts = np.unique(comp_list, return_counts=True)
    dictio=dict(zip(unique, counts))
    majority_comp = max(dictio, key=dictio.get)
    mask=np.zeros((width,height))
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            mask[x,y]=int(comp==majority_comp)
    return mask

def get_green_mask(image_array, hue_range=(35, 90)):
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    hue_channel = hsv_image[:, :, 0]
    lower_hue, upper_hue = hue_range
    green_mask = cv2.inRange(hue_channel, lower_hue, upper_hue)
    return green_mask