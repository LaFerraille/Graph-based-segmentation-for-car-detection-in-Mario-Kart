# Graph-based Segmentation for Car Detection in Mario Kart

This repository contains code and resources for a project on car detection in Mario Kart using graph-based algorithms. The project was conducted as part of the Graphical Models course at CentraleSupélec. 

## Introduction

Image segmentation is a critical task in computer vision, particularly in scenarios like detecting cars in Mario Kart gameplay footage. This project explores the application of graph-based algorithms for segmenting cars in Mario Kart images and videos. Two main approaches were investigated: a tree-based method with masking and the Max-Flow/Min-Cut algorithm.

## Tree-based Method with Masking

The tree-based method, originally proposed by Felzenszwalb et al. [1], segments an image by representing it as an undirected graph. The algorithm iteratively merges components based on edge weights until a segmentation is obtained. We implemented this method and enhanced it with masking techniques to improve car detection in Mario Kart images.

## Max-Flow/Min-Cut Algorithm

The Max-Flow/Min-Cut algorithm, introduced by Boykov and Kolmogorov [2], partitions a graph into two disjoint sets by finding the minimum cut. We adapted this algorithm for car detection by transforming the image into a graph and applying manual seeds for object and background pixels.

## Results

The methods were evaluated on a set of 15 annotated images extracted from a 1 minute video containing various situations (normal, storm, tiny, bananas, ink...) using the Mean Intersection over Union (MIoU) metric:

$$
\text{MIoU}(P,GT) = \frac{\sum_{i=1}^{N} \frac{|P_i \cap GT_i|}{|P_i \cup GT_i|}}{N}
$$

where $P$ are the predicted masks and $GT$ are the ground truths.

As shown in Table \ref{tab:miuo_results}, the two methods give similar results, with a slightly better performance for the Boykov-Kolmogorov method (higher MIoU and smaller variance).

\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|}
        \hline
        Method & mIoU \\
        \hline
        Tree Based (TB) & $0.58 \pm 0.24$ \\
        Boykov-Kolmogorov (BK) & $0.59 \pm 0.21$ \\
        \hline
    \end{tabular}
    \caption{Mean Intersection over Union (MIoU) on 15 annotated images}
    \label{tab:miuo_results}
\end{table}

## Usage

To use the code in this repository:

1. Clone the repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the scripts or notebooks provided either in `MaxFlow-MinCut` or `Tree-based-clustering` directory to perform car detection on your Mario Kart images or videos.

## References

[1] [P. F. Felzenszwalb and D. P. Huttenlocher, "Efficient Graph-Based Image Segmentation," *International Journal of Computer Vision*, vol. 59, no. 2, pp. 167-181, 2004.](https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf)

[2] [Y. Boykov and V. Kolmogorov, "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 26, no. 9, pp. 1124-1137, 2004.](https://www.csd.uwo.ca/~yboykov/Papers/pami04.pdf)

## Contributors

- Chiara Roverato
- Quentin Gopée
- Raphaël Romand-Ferroni

For any questions or inquiries, please contact [Raphaël Romand-Ferroni](mailto:raphael.romandferroni@student-cs.fr) [Chiara Roverato](mailto:chiara.roverato@student-cs.fr) or [Quentin Gopée](mailto:quentin.gopee@student-cs.fr).
