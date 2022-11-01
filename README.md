# Dynamic Time Warping (DTW)
The code written is implements the Dynamic Time Warping algorithm from scratch to measure the similarity between two temporal sequences. 
Specifically, the code was written to apply similarity measures on a heartbeat time series dataset and a molecular graphs dataset.

The code in the script includes the Python function named for implementing the DTW distance function with constrained warping: constrained_dtw(t1, t2, w)

Note:
- t1 and t2, are both lists of type float, while the argument w is a non-negative integer not smaller than the difference in length between the two time series t1 and t2 to be
compared
- Manhattan distance is used as the base elementwise distance function for DTW
- The value returned is a float
