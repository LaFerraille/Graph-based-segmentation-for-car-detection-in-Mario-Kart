Banana Detection in Mario Kart using Graph Cut Algorithm
==============================

Graph Cut Algorithm: Implement a graph cut algorithm tailored for segmenting bananas in Mario Kart images. Explore variations such as max-flow/min-cut algorithms for efficient segmentation.


- [Efficient Graph-Based Image Segmentation](https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf) : Simple Linear Iterative Clustering (SLIC) Algorithm, addresses the problem of segmenting an image into regions, graph-based approach to segmentation.  Let $G = (V,E)$ be an undirected graph with vertices $v_i \in V$ , the set of elements to be segmented, and edges $(v_i, v_j ) \in E$ corresponding to pairs of neighboring vertices. Each edge $(v_i, v_j ) \in E$ has a corresponding weight $w((v_i,v_j))$, which is a non-negative measure of the dissimilarity between neighboring elements $v_i$ and $v_j$. In the case of image segmentation, the elements in V are pixels and the weight of an edge is some measure of the dissimilarity between the two pixels connected by that edge (e.g., the difference in intensity, color, motion, location or some other local attribute).