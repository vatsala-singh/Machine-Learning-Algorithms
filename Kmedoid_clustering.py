#!/usr/bin/env python
# coding: utf-8

# Kmedoid clustering: 
# 
# 1. K-medoids are a prominent clustering algorithm as an improvement of the predecessor, the K-Means algorithm. Despite its being widely used and less sensitive to noises and outliers, the performance of the K-medoids clustering algorithm is affected by the distance function
# 
# 2. k-means algorithm is not appropriate to make objects of the cluster to the data points then the k-medoid clustering algorithm is preferred
# 
# 3. A medoid is an object of the cluster whose dissimilarity to all the objects in the cluster is minimum
# 
# 4. Kmeans - Euclidian Distance, Kmedoid -work with an arbitrary matrix
# 
# 5. is a classical partitioning technique of clustering that cluster the dataset into a k cluster. It is more robust to noise and outliers because it may minimize the sum of pair-wise dissimilarities however k-means minimize the sum of squared Euclidean distances
# 
# 6. The most common distances used in KMedoids clustering techniques are Manhattan distance or Minkowski distance and here we will use Manhattan distance.
# 

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing


# In[64]:


class Kmedoid:
    #initialise
    def __init__(self,data,k,iters):
        self.data= data
        self.k = k
        self.iters = iters
        self.medoids = np.array([data[i] for i in range(self.k)])
        self.colors = np.array(np.random.randint(0, 255, size =(self.k, 4)))/255
        self.colors[:,3]=1
        
    #calculate manhattan distance
    def manhattan(self, p1, p2):
        return np.abs((p1[0]-p2[0])) + np.abs((p1[1]-p2[1]))
    
    #calculate cost function to find the minimum cost between data points and medoids
    def cost_fun(self, medoids):
        t_cluster = {i:[] for i in range(len(medoids))}
        net_cost = 0
        for d in self.data:
            distance = np.array([self.manhattan(d, md) for md in medoids])
            cost = distance.argmin()
            t_cluster[cost].append(d)
            net_cost +=distance.min()
            
        t_cluster = {k:np.array(v) for k,v in t_cluster.items()}
        return t_cluster, net_cost
    
    def fit(self):
        samples,_ = self.data.shape
        self.clusters, cost = self.cost_fun(self.medoids)
        count = 0
        
        colors =  np.array(np.random.randint(0, 255, size =(self.k, 4)))/255
        colors[:,3]=1
        
        plt.title(f"Step : 0")
        [plt.scatter(self.clusters[t][:, 0], self.clusters[t][:, 1], marker=".", s=100,
                                        color = colors[t]) for t in range(self.k)]
        plt.scatter(self.medoids[:, 0], self.medoids[:, 1], s=200, color=colors)
        plt.show()
        
        #for each medoid point m and non medoid point n
        # swam m and n and. calculate the cost
        #if cost is increased from previous step, undo the swap
        while True:
            swap = False
            for i in range(samples):
                if not i in self.medoids:
                    for j in range(self.k):
                        tmp_meds = self.medoids.copy()
                        tmp_meds[j] = i
                        clusters_, cost_ = self.cost_fun(medoids=tmp_meds)

                        if cost_<cost:
                            self.medoids = tmp_meds
                            cost = cost_
                            swap = True
                            self.clusters = clusters_
                            print(f"Medoids Changed to: {self.medoids}.")
                            plt.title(f"Step : {count+1}")  
                            [plt.scatter(self.clusters[t][:, 0], self.clusters[t][:, 1], marker=".", s=100,
                                        color = colors[t]) for t in range(self.k)]
                            plt.scatter(self.medoids[:, 0], self.medoids[:, 1], s=200, color=colors)
                            plt.show() 
                            
            count+=1

            if count>=self.iters:
                print("End of the iterations.")
                break
            if not swap:
                print("No changes.")
                break
        return self.clusters


# In[49]:


dataset = dataset = pd.read_csv('cancer.csv')


# In[4]:


dataset_cluster = dataset.drop(['id','diagnosis','Unnamed: 32'], axis = 1)


# In[7]:


scaler = preprocessing.MinMaxScaler()
data_scaled = scaler.fit_transform(dataset_cluster)
X = data_scaled


# In[65]:


kmed = Kmedoid(X, 2, 100)
y_pred = kmed.fit()


# In[61]:


len(y_pred[0])


# In[52]:


len(y_pred[1])


# In[ ]:




