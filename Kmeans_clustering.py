#!/usr/bin/env python
# coding: utf-8

# Kmeans Clustering:
# 
# 1. Unsupervised Learning
# 2. No target variable, find pattern and group them into clusters
# 
# Intuition:
# 
# - Segeregate n data points into k clusters
# - based on centroid and euclidian distance
# 
# Algorithm:
# 
# 1. Randomly initialize cluster centers of each cluster from the data points
# 2. 
#     a. For each data points compute euclidian distance from all centroids
#     b. Adjust centroids of each cluster by taking the average of all data points whcih belongs to the cluster in 2a, and move the centroid to mean
#     
# 3. Repeat the process till clusters are well separated or convergence is achieved
# 
# 
# https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42
# 
# 

# In[87]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# In[ ]:


class KmeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iter = 100
        self.num_examples, self.num_features = X.shape
        self.plot_figure = True
        
    def init_random_centroids(self, X):
        #centroid with K*N dimension: 2*31
        centroids = np.zeros((self.K,self.num_features))
        for k in range(self.K):
            #assigning random centroids
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid
        return centroids
    
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        for p_idx, p in enumerate(X):
            c_centroid = np.argmin(np.sqrt(np.sum((p-centroids)**2, axis=1)))
            clusters[c_centroid].append(p_idx)
        return clusters
    
    def new_centroid(self, cluster, X):
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(cluster):
            new_c = np.mean(X[cluster], axis =0)
            centroids[idx]=new_c
        return centroids
    
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)
        for c_idx, c in enumerate(clusters):
            for sample_idx in c:
                y_pred[sample_idx] = c_idx
        return y_pred
        
    def plot_fig(self, X, y):
        fig = px.scatter(X[:, 1], X[:, 2], color=y)
        fig.show() # visualize
        
    def fit(self, X):
        #initiate random centroids
        centroids = self.init_random_centroids(X)
        #create clusters
        for _ in range(self.max_iter):
            clusters = self.create_cluster(X, centroids)
            prev_centroids = centroids
            centroids = self.new_centroid(clusters, X)
            diff = centroids-prev_centroids
            
            if not diff.any():
                break
            y_pred = self.predict_cluster(clusters, X)
            
            if self.plot_figure: # if true
                self.plot_fig(X, y_pred) # plot function 
            
        return y_pred
    


# In[56]:


dataset = dataset = pd.read_csv('cancer.csv')


# In[57]:


dataset_cluster = dataset.drop(['id','diagnosis'], axis = 1)


# In[58]:


X = dataset_cluster.to_numpy()


# In[59]:


X.shape


# In[60]:


n_cluster = 2


# In[68]:





# In[62]:


dataset_cluster['cluster'] = y_pred


# In[63]:


dataset_cluster['cluster'].value_counts()


# In[33]:


from sklearn import preprocessing


# In[35]:


dataset['diagnosis'].value_counts()


# In[38]:


dataset_cluster


# In[37]:


scaler = preprocessing.MinMaxScaler()
data_scaled = scaler.fit_transform(dataset_cluster)
X = data_scaled


# In[64]:


dataset_cluster = dataset_cluster.drop(['Unnamed: 32'], axis=1)


# In[65]:


scaler = preprocessing.MinMaxScaler()
data_scaled = scaler.fit_transform(dataset_cluster)
X = data_scaled


# In[66]:


X


# In[72]:


kmeans = KmeansClustering(X, n_cluster)
y_pred = kmeans.fit(X)


# In[69]:


dataset_cluster['cluster'] = y_pred


# In[73]:


dataset_cluster['cluster'].value_counts()


# In[ ]:





# In[85]:


color=['red','blue']
labels=['cluster1','cluster2']


colors = {0:'red', 1:'green'}
grouped = dataset_cluster.groupby('cluster')

fig, ax = plt.subplots()


for key,group in grouped:
    group.plot(ax=ax, kind = 'scatter', x = 'radius_mean', y= 'texture_mean', label = key, color = colors[key])
    
plt.show()


# In[ ]:





# In[ ]:




