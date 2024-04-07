# Graph-based Segmentation for Car Detection in Mario Kart

==========================================================

This repository contains code and resources for a project on car detection in Mario Kart using graph-based algorithms. The project was conducted as part of the Graphical Models course at CentraleSupélec. 

## Introduction

Image segmentation is a critical task in computer vision, particularly in scenarios like detecting cars in Mario Kart gameplay footage. This project explores the application of graph-based algorithms for segmenting cars in Mario Kart images and videos. Two main approaches were investigated: a tree-based method with masking and the Max-Flow/Min-Cut algorithm.

## Tree-based Method with Masking

The tree-based method, originally proposed by Felzenszwalb et al. [1], segments an image by representing it as an undirected graph. The algorithm iteratively merges components based on edge weights until a segmentation is obtained. We implemented this method and enhanced it with masking techniques to improve car detection in Mario Kart images.

## Max-Flow/Min-Cut Algorithm

The Max-Flow/Min-Cut algorithm, introduced by Boykov and Kolmogorov [2], partitions a graph into two disjoint sets by finding the minimum cut. We adapted this algorithm for car detection by transforming the image into a graph and applying manual seeds for object and background pixels.

## Results

The project yielded promising results in segmenting cars in Mario Kart images and videos. We observed significant improvements in car detection accuracy compared to traditional methods.

## Usage

To use the code in this repository:

1. Clone the repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the scripts or notebooks provided in the `src` directory to perform car detection on your Mario Kart images or videos.

## References

[1] [P. F. Felzenszwalb and D. P. Huttenlocher, "Efficient Graph-Based Image Segmentation," *International Journal of Computer Vision*, vol. 59, no. 2, pp. 167-181, 2004.](https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf)

[2] [Y. Boykov and V. Kolmogorov, "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 26, no. 9, pp. 1124-1137, 2004.](https://www.csd.uwo.ca/~yboykov/Papers/pami04.pdf)

## Contributors

- Chiara Roverato
- Quentin Gopée
- Raphaël Romand-Ferroni

For any questions or inquiries, please contact [Raphaël Romand-Ferroni](mailto:raphael.romandferroni@student-cs.fr) [Chiara Roverato](mailto:chiara.roverato@student-cs.fr) or [Quentin Gopée](mailto:quentin.gopee@student-cs.fr).
