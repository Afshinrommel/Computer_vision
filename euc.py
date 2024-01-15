from scipy.spatial import distance as dist 
from configs import config



def eucli_dist(persons):
 violate = set()
 if len(persons) >= 2:
        # extract all centroids from the results and compute the Euclidean distances
        # between all pairs of the centroids
#        centroids = np.array([r[2] for r in results])
        D = dist.cdist(persons, persons, metric="euclidean")
        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                # check to see if the distance between any two centroid pairs is less
                # than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)
 return(violate)