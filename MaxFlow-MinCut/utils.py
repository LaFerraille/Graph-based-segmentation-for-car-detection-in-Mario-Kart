import cv2
import math
import numpy as np


def draw_sp_mask(I,SP):
	I_marked=np.zeros(I.shape)
	I_marked=np.copy(I)
	mask=SP.getLabelContourMask()
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j]==-1 or mask[i][j]==255: # SLIC/SLICO marks borders with -1 :: SEED marks borders with 255
				I_marked[i][j]=[128,128,128]
	return I_marked


def draw_centroids(I, SP_list):
	for each in SP_list:
		i,j=each.centroid
		I[i][j]=128
	return I


def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def apply_mask(image, G, G_residual):
	h , w, _ = image.shape
	St, _ = G_residual.graph['trees']
	partition = (set(St), set(G) - set(St))
	mask = image.copy() // 4
	
	for sp in partition[0]:
		for pixels in sp.pixels:
			i, j = pixels
			mask[i][j] = image[i][j]
			
	# image_mask = cv2.bitwise_and(image, image, mask=mask)
	return mask


def get_mask(image, G, G_residual):
    h , w, _ = image.shape
    St, _ = G_residual.graph['trees']
    partition = (set(St), set(G) - set(St))
    mask = np.zeros((h, w), np.uint8)
    
    for sp in partition[0]:
        for pixels in sp.pixels:
            i, j = pixels
            mask[i][j] = 1
            
    return mask


def process_mask(mask, threshold):
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    output = np.zeros_like(mask, dtype=np.uint8)
    region_sizes = np.bincount(labels.ravel())[1:]

    for label in range(1, num_labels):
        if region_sizes[label-1] >= threshold:
            output[labels == label] = 1
            
    return output