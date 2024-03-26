import cv2
import math
import numpy as np


def mark_seeds(event,x,y,flags,param):
    global drawing,mode,marked_bg_pixels,marked_ob_pixels,I_dummy
    h,w,c=I_dummy.shape

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == "ob":
                if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
                    marked_ob_pixels.append((y,x))
                cv2.line(I_dummy,(x-3,y),(x+3,y),(0,0,255))
            else:
                if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
                    marked_bg_pixels.append((y,x))
                cv2.line(I_dummy,(x-3,y),(x+3,y),(255,0,0))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == "ob":
            cv2.line(I_dummy,(x-3,y),(x+3,y),(0,0,255))
        else:
            cv2.line(I_dummy,(x-3,y),(x+3,y),(255,0,0))


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