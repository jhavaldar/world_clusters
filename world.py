import requests
import re
import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from world_bank_api import get_country_from_code, get_name_of_indicator
import matplotlib.pyplot as plt
from math import sqrt

final_features = "IP.JRN.ARTC.SC,BM.GSR.MRCH.CD,EG.EGY.PRIM.PP.KD,SH.ALC.PCAP.LI,EN.ATM.METH.EG.KT.CE,SP.URB.TOTL,NY.GDP.TOTL.RT.ZS,SP.URB.GROW,BM.GSR.CMCP.ZS,EP.PMP.DESL.CD,SP.URB.TOTL.IN.ZS"

# Regularizes features.
def regularize(inpath, outpath):
  df = pd.read_csv(inpath)
  # For each column that is not names
  col_labels = list(df.columns)
  for label in col_labels[2:]:
    abs_max = 0
    column = list(df[label])
    min = abs(np.min(column))
    max = abs(np.max(column))
    #print min, max
    if min > max:
      abs_max = min
    else:
      abs_max = max
    #print column
    new_column = [(entry+0.0)/abs_max for entry in column]
    df[label] = new_column
  col_labels = list(df.columns)
  del df[col_labels[0]]
  df.to_csv(outpath, index=False)

# Returns two dimensional projection of dataframe using PCA
def pca_projection(inpath="extract_clusters.csv", outpath="projection.csv"):
  df = pd.read_csv(inpath)
  pca = PCA(n_components=2)
  names = df['name']
  del df['name']
  clusters = df['cluster']
  del df['cluster']
  df = pca.fit_transform(df)
  df = pd.DataFrame(df)
  df['name'] = names
  df['cluster'] = clusters
  df.to_csv(outpath, index=False)

# Re-organizes the feature clusters using PCA
def trans_pca(inpath="trans_nonames.csv", outpath="pca_comps.csv"):
  df = pd.read_csv(inpath)
  labels = list(df.columns)
  names = list(df[labels[0]])
  del df[labels[0]]
  pca = PCA()
  pca.fit(df)
  variances = list(pca.explained_variance_ratio_ )

  sum = 0
  n_comp = 0
  while(sum < 0.85):
    sum += variances[n_comp]
    n_comp += 1
  print n_comp

  pca = PCA(n_components=n_comp)
  df = pca.fit_transform(df)
  df = pd.DataFrame(df)
  df['name'] = names
  df.to_csv(outpath, index=False)

# Returns the transpose of a dataframe; instead of countries along features,
# plot features along countries
def get_transpose(inpath="full_data_reg.csv", outpath="trans_nonames.csv"):
  df = pd.read_csv(inpath)
  labels = list(df.columns)
  del df['name']
  new_df = df.transpose()
  new_df.to_csv(outpath, index=False)
  return new_df

# Euclidean distance between two vectors
def dist(a, b):
  if len(a) <> len(b):
    return -1
  else:
    sum = 0
    for i in range(0, len(a)):
      sum += (a[i] - b[i])**2
    return sqrt(sum)

# For a dataframe consisting of clustered features
#  compute its score, i.e. the average distance from each point to the centroid of its cluster
def score(df):
  labels = list(df.columns)
  labels = [label for label in labels if label not in ['name','cluster']]
  # Group the data into clusters of points
  clusters = {}
  for index, row in df.iterrows():
    point = [row[label] for label in labels]
    point = np.array(point)
    cluster = row["cluster"]
    if cluster in clusters:
      clusters[cluster].append(point)
    else:
      clusters[cluster] = [point]
  num_points =  df.shape[0] + 0.0
  # Get the sum of squared errors
  sse = 0
  for label in clusters:
    centroid = []
    group = clusters[label]
    length = len(group) + 0.0
    # Compute each coordinate of the centroid
    for index in range(0, len(group[0])):
      sum = np.sum(np.array([point[index] for point in group]))/length
      centroid.append(sum)
    # Add the sum squared errors for the cluster
    for index in range(0, len(group)):
      sse += dist(centroid, group[index])**2
  return sse

# Create the plot of sum of squared errors vs k to find the "best" k for k means
def elbow_plot(inpath="pca_comps.csv", n_max=10):
  df = pd.read_csv(inpath)
  labels = list(df.columns)

  #Delete the feature names
  del df['name']

  scores = []
  for n in range(1,n_max):
    k_means = cluster.KMeans(n_clusters=n)
    #Fit the k means to our test data
    k_means.fit(df)
    labels = list(k_means.labels_)  #The labels
    df['cluster'] = labels
    k_score = score(df)
    scores.append(k_score)

  k_range = [i for i in range(1,n_max)]
  plt.plot(k_range, scores, alpha=0.5)
  plt.title("Score vs. number of clusters")
  plt.ylabel('Score')
  plt.xlabel('k')
  plt.savefig('clusters.png')
  plt.show()

# Group into clusters and draw the clusters
def create_clusters(inpath="pca_comps.csv", outpath="feature_cluster_comp.csv", n=3):
  df = pd.read_csv(inpath)
  labels = list(df.columns)

  #Delete the feature names
  names = df['name']
  del df['name']

  k_means = cluster.KMeans(n_clusters=n)
  #Fit the k means to our test data
  k_means.fit(df)
  labels = list(k_means.labels_)  #The labels
  df['cluster'] = labels
  df['name'] = names
  df.to_csv(outpath, index=False)
  return df.as_matrix()

# Draw clusters craeted from k-means
def draw_clusters(inpath="named_feature_cluster_comp.csv"):
  colors = ["red", "blue", "orange"]
  df = pd.read_csv(inpath)
  x = df['0']
  y = df['1']
  colors = [colors[int(i)] for i in df['cluster']]
  indic_names = df['indicator_name']
  plt.scatter(x, y, c=colors)
  for i in range(0,len(x)):
    plt.annotate(indic_names[i], (x[i],y[i]))
  plt.savefig('clusters_labeled.png')
  plt.show()

# Given a set of indicators, add on the full names as a column
def indic_names(inpath="feature_cluster_comp.csv", outpath="named_feature_cluster_comp.csv"):
  df = pd.read_csv(inpath)
  indics = list(df['name'])
  full_names = [get_name_of_indicator(indic) for indic in indics]
  df['indicator_name'] = full_names
  df.to_csv(outpath, index=False)
  print "Done"

# Creates an elbow plot for k-means on nations with restricted feature set
def elbow_plot_nations(feats, inpath="full_data_reg.csv", outpath="full_data_extract.csv"):
  features = feats.split(",")
  features.append('name')
  df = pd.read_csv(inpath)
  # Get the relevant features into our dataframe
  df = df[features]
  # Store the names of countries for later
  names = df['name']
  df.to_csv(outpath, index=False)
  # Use the elbow plot to determine the best number 
  elbow_plot(inpath=outpath,n_max=10)

def get_nation_names(inpath="extract_clusters.csv", outpath="final_clusters.csv"):
  df = pd.read_csv(inpath)
  names = [get_country_from_code(str(code)) for code in list(df['name'])]
  df['full_name'] = names
  df.to_csv(outpath, index=False)

def final_plot(inpath="final_clusters.csv"):
  colors = ["red", "blue", "green", "orange"]
  df = pd.read_csv(inpath)
  x = df['0']
  y = df['1']
  names = df['full_name']
  colors = [colors[int(i)] for i in df['cluster']]
  plt.scatter(x, y, c=colors, alpha=0.5)
  plt.savefig('final_clusters.png')
  plt.show()

#trans_pca()
#elbow_plot()
#create_clusters()
#indic_names()
#draw_clusters()


#elbow_plot_nations(final_features)
create_clusters(inpath="full_data_extract.csv", outpath="nation_clusters2.csv", n=4)
pca_projection(inpath="nation_clusters2.csv", outpath="projection.csv")
get_nation_names(inpath="projection.csv")