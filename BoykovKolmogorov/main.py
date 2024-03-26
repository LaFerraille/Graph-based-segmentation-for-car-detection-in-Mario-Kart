import sys, getopt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from graph_cut import BoykovKolmorogov
from img_to_graph import img_to_graph
from GUI import GUI_seeds


def main():
	inputfile = ''
	try:
		opts, args = getopt.getopt(sys.argv[1:], "i:h", ["input-image=", "help"])
	except getopt.GetoptError:
		print('fast_seg.py -i <input image>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print ('fast_seg.py -i <input image>')
			sys.exit()
		elif opt in ("-i", "--input-image"):
			inputfile = arg
	print('Using image: ', inputfile)

	image = cv2.imread(inputfile)
	
	GUI = GUI_seeds(inputfile)
	marked_ob_pixels, marked_bg_pixels = GUI.labelling()

	G, s, t, I_marked, sp_lab = img_to_graph(image, marked_ob_pixels, marked_bg_pixels)
	G_residual = BoykovKolmorogov(G, s, t, capacity='sim').max_flow()
	
	h , w, _ = image.shape
	St, _ = G_residual.graph['trees']
	partition = (set(St), set(G) - set(St))
	F=np.zeros((h,w),dtype=np.uint8)
	for sp in partition[0]:
		for pixels in sp.pixels:
			i, j = pixels
			F[i][j] = 1
	Final = cv2.bitwise_and(image, image, mask=F)


	plt.subplot(2,2,1)
	plt.tick_params(labelcolor='black', top='off', bottom='off', left='off', right='off')
	plt.imshow(image[...,::-1])
	plt.axis("off")
	plt.xlabel("Input image")

	plt.subplot(2,2,2)
	plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
	plt.imshow(I_marked[...,::-1])
	plt.axis("off")
	plt.xlabel("Super-pixel boundaries and centroid")

	plt.subplot(2,2,3)
	plt.imshow(sp_lab)
	plt.axis("off")
	plt.xlabel("Super-pixel representation")

	plt.subplot(2,2,4)
	plt.imshow(Final[...,::-1])
	plt.axis("off")
	plt.xlabel("Output Image")
	
	cv2.imwrite("out.png",Final)
	plt.show()

if __name__ == '__main__':
	main()