from sklearn.cluster import KMeans
import numpy as np

def distance(p1,p2):
  res = 0
  for i in range(len(p1)):
    res = res + (p1[i] - p2[i])**2
  return res**(0.5)

def maxDistance(a):
    aux = []
    for x in a:
        aux.append(maxDist(x,a))
    return aux

def maxDist(a,all):
    aux = []
    for x in all:
        if not np.array_equal(x,a):
            aux.append(distance(a,x))
    return max(aux)

def std(a):
    dis = maxDistance(a)
    m = (2*len(a))**0.5
    aux = []
    for i in range(len(dis)):
        aux.append(dis[i]/m)
    return aux

def k_means(labels, numCluster):
    kmeans = KMeans(n_clusters=numCluster, random_state=0).fit(labels)
    a = kmeans.cluster_centers_
    return kmeans.cluster_centers_,std(a)