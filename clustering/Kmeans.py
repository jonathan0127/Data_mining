# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:45:53 2025

@author: HP
"""

import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt

#Randomly choose k data as initial centroids
def initial_centroid(samples ,k):
    if k > len(samples):
        raise ValueError(f"Cannot initialize {k} centroids with only {len(samples)} samples.")
    indices=np.random.choice(len(samples),k,replace=False)
    if hasattr(samples, 'values'):
        samples = samples.values
    return samples[indices]

#Calculate the distances matrix between whole samples and every centroids
def cal_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))
#Calculate the sum of moving shift of every centroids
def cal_centroid_shift(old_centroids, new_centroids):
    shifts = np.sqrt(np.sum((old_centroids-new_centroids)**2))
    return np.sum(shifts)
#Assign samples to clusters that they belong
def assign_cluster(samples, centroids,k):
   cluster_assign=[]
   for sample in samples:
       min_distance=cal_distance(sample, centroids[0])
       cluster_num=0 #Record the cluster number that this sample belongs to 
       for i in range(k-1):
           #calculate the distance between sample and centroid
           distance=cal_distance(sample, centroids[i+1])
           if distance<min_distance:
               min_distance=distance #update the smallest distance
               cluster_num=i+1             
       cluster_assign.append(cluster_num)
   return np.array(cluster_assign)

#Update the centroids by calculating every clusters' mean values   
def update_centroid(samples, old_centroids, assigned_cluster, k):
    if hasattr(samples, 'values'):
        samples = samples.values
    
    new_centroids = np.zeros_like(old_centroids)
    for i in range(k):
        cluster_points=samples[assigned_cluster==i]
        #If the ith cluster gas no samples, randomly choose a sample for new centroid
        if len(cluster_points)==0:
            min_dist=np.full(len(samples),np.inf)  #Initialized the min distance as infinite large
            for j in range(len(samples)):
                dist=cal_distance(samples[i], old_centroids)
                if dist<min_dist[i]:
                    min_dist[i]=dist
            #Find the farthest(min_dist is the largest one) point
            farthest_index=np.argmax(min_dist)
            new_centroids[i]=samples[farthest_index]
        else: #update the centroid if cluster is not empty
            new_centroids[i]=cluster_points.mean(axis=0)
    return new_centroids
  
def kmeans(samples, k, cutoff, max_iter):      
    #Create a list to store centroids in every cluster
    centroids=initial_centroid(samples, k)
    loop_num=0 
    while True:
        assigned_cluster=assign_cluster(samples, centroids, k)
        new_centroids=update_centroid(samples, centroids, assigned_cluster, k)
        #Calculate the distance between new centroid and old centroid
        shift=cal_centroid_shift(centroids, new_centroids)
        #See if convergence(less than cutoff: convergence, else: no convergence)
        if shift<cutoff:
            print(f"In the {loop_num+1}th loop, k-means converged")
            break  
        centroids=new_centroids 
        loop_num+=1
        #If no convergence happens before reaching maximum iteration, end the loop
        if loop_num>=max_iter:
            print("No convergence happens before reaching max iteration")
            break          
    return assigned_cluster, centroids

#Calculate the inertia: the square distance between every sample and the cluster centroid that it belongs to 
def cal_inertia(samples, centroids, labels):
    inertia=0.0
    for i in range(len(samples)):
        center=centroids[labels[i]]
        inertia+=np.sum((samples[i]-center)**2)
    return inertia

# elbow method
def find_best_k(samples, cutoff, max_iter ,max_k=10,plot_path=None):
    #Check if the sample data has NaN or inf(Missing), if yes, process the samples
    if np.isnan(samples).any() or np.isinf(samples).any():
        samples=np.nan_to_num(samples, nan=np.nanmean(samples), posinf=np.nanmax(samples), neginf=np.nanmin(samples))
    #A lists to store internias
    inertias=[]
    n_samples=len(samples)
    #Ensure that k won't over number of samples
    max_k=min(max_k, n_samples)
    k_range=range(1,max_k+1)
    for k in k_range:
        if len(samples) < k:
            print(f"[Warning] Skipping k={k}, not enough samples ({len(samples)} < {k})")
            continue
        labels, centroids = kmeans(samples, k, cutoff, max_iter)
        inertia=cal_inertia(samples, centroids, labels)
        #filter NaN inertia values
        if np.isnan(inertia) or np.isinf(inertia):
            print(f"[Warning] Inertia at k={k} is invalid (NaN or inf), skipping.")
            continue
        inertias.append(inertia)
    if len(inertias)<=1:
        return 1
    #Use kneed to find out elbow
    kl=KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    best_k=kl.elbow or 1  
    #Draw a graph
    plt.figure()
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Best k')
    if kl.elbow:
        plt.axvline(x=kl.elbow, color='red', linestyle='--', label=f'Best k = {kl.elbow}')
        plt.legend()
    plt.grid(True)
    if plot_path:
        plt.savefig(plot_path, bbox_inches='tight')
    # plt.show()    
    return best_k

class KMeansCluster:
    def __init__(self, k, cutoff=1e-3, max_iter=100):
        self.k = k
        self.cutoff = cutoff
        self.max_iter = max_iter
        self.centroids = None
        self.labels_ = None

    def fit(self, samples):
        self.labels_, self.centroids = kmeans(samples, self.k, self.cutoff, self.max_iter)
        return self.labels_