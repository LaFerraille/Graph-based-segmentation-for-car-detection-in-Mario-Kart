import networkx as nx
import math
import cv2
import numpy as np
from utils import distance


class SPNode():
	def __init__(self, label=None, pixels=[], mean_intensity=0.0, centroid=(), type='na', mean_lab=None, lab_hist=None, real_lab=None):
		self.label = label
		self.pixels = pixels
		self.mean_intensity = mean_intensity
		self.centroid = centroid
		self.type = type
		self.mean_lab = mean_lab
		self.lab_hist = lab_hist
		self.real_lab = real_lab
	def __repr__(self):
		return str(self.label)
     

def gen_sp(image, pixel_obj, pixel_bg, algorithm=101, region_size=20, num_iter=4):
    SP = cv2.ximgproc.createSuperpixelSLIC(image, algorithm=algorithm, region_size=region_size, ruler=10.0)
    SP.iterate(num_iterations=num_iter)

    SP_labels=SP.getLabels()
    SP_list=[None for each in range(SP.getNumberOfSuperpixels())]

    h, w, _ = image.shape
    I_lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)

    for i in range(h):
        for j in range(w):
            if not SP_list[SP_labels[i][j]]:
                tmp_sp = SPNode(label=SP_labels[i][j], pixels=[(i,j)])
                SP_list[SP_labels[i][j]]=tmp_sp
            else:
                SP_list[SP_labels[i][j]].pixels.append((i,j))

    for sp in SP_list:
        n_pixels = len(sp.pixels)
        i_sum = 0
        j_sum = 0
        lab_sum = [0,0,0]
        tmp_mask = np.zeros((h,w),np.uint8)
        for each in sp.pixels:
            i,j = each
            i_sum += i
            j_sum += j
            lab_sum = [x + y for x, y in zip(lab_sum, I_lab[i][j])]
            tmp_mask[i][j] = 255
        sp.lab_hist = cv2.calcHist([I_lab], [0,1,2], tmp_mask, (32,32,32), [0, 255, 0, 255, 0, 255])
        sp.centroid += (i_sum//n_pixels, j_sum//n_pixels,)
        sp.mean_lab = [x/n_pixels for x in lab_sum]
        sp.real_lab = [sp.mean_lab[0]*100/255, sp.mean_lab[1]-128, sp.mean_lab[2]-128]

    # Label the marked pixels
    for pixels in pixel_obj:
        x,y = pixels
        SP_list[SP_labels[x][y]].type="ob"
    for pixels in pixel_bg:
        x,y = pixels
        SP_list[SP_labels[x][y]].type="bg"
    mask_ob=np.zeros((h,w),dtype=np.uint8)
    for pixels in pixel_obj:
        i,j=pixels
        mask_ob[i][j]=255
    mask_bg=np.zeros((h,w),dtype=np.uint8)
    for pixels in pixel_bg:
        i,j=pixels
        mask_bg[i][j]=255

    # Get the histograms
    hist_ob=cv2.calcHist([image],[0,1,2],mask_ob,(32,32,32),[0, 255, 0, 255, 0, 255])
    hist_bg=cv2.calcHist([image],[0,1,2],mask_bg,(32,32,32),[0, 255, 0, 255, 0, 255])

    return SP, SP_list, hist_ob, hist_bg   
	

def gen_graph(I, SP_list, hist_ob, hist_bg, bins=(32,32,32), lambda_=0.9, sigma=5):
    G = nx.Graph()
    s = SPNode(label='s')
    t = SPNode(label='t')
    hist_ob_sum = int(hist_ob.sum())
    hist_bg_sum = int(hist_bg.sum())

    for u in SP_list:
        K=0
        region_rad=math.sqrt(len(u.pixels)/math.pi)
        for v in SP_list:
            if u != v:
                if distance(u.centroid, v.centroid) <= 2.5*region_rad:
                    sim = math.exp(-(cv2.compareHist(u.lab_hist,v.lab_hist,3)**2/2*sigma**2))*(1/distance(u.centroid, v.centroid))
                    K += sim
                    G.add_edge(u, v, sim=sim)
        if(u.type=='na'):
            l_,a_,b_ = [int(x) for x in u.mean_lab]
            l_i = int(l_//(255/bins[0]))
            a_i = int(a_//(255/bins[1]))
            b_i = int(b_//(255/bins[2]))
            pr_ob = int(hist_ob[l_i,a_i,b_i])/hist_ob_sum
            pr_bg = int(hist_bg[l_i,a_i,b_i])/hist_bg_sum
            sim_s = 100000
            sim_t = 100000
            if pr_bg > 0:
                sim_s = lambda_*-np.log(pr_bg)
            if pr_ob > 0:
                sim_t = lambda_*-np.log(pr_ob)
            G.add_edge(s, u, sim=sim_s)
            G.add_edge(t, u, sim=sim_t)
        if(u.type=='ob'):
            G.add_edge(s, u, sim=1+K)
            G.add_edge(t, u, sim=0)
        if(u.type=='bg'):
            G.add_edge(s, u, sim=0)
            G.add_edge(t, u, sim=1+K)		
    return G, s, t