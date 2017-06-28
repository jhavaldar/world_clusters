import requests
import re
import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from world_bank_api import get_country_from_code, get_name_of_indicator
import matplotlib.pyplot as plt

def transmeans(inpath = "transpose.csv", outpath = "transclusters.csv", n=5):
  k_means = cluster.KMeans(n_clusters=n)
  df = pd.read_csv(inpath)
  test = pd.read_csv(inpath)
  col_labels = list(test.columns)
  del test[col_labels[0]]
  #Regularize
  for col in list(test.columns):
    bound = np.mean(list(test[col]))
    regs = [elt/bound for elt in list(test[col])]
    test[col] = regs
    df[col] = regs

  #Fit the k means to our test data
  k_means.fit(test)
  labels = list(k_means.labels_)  #The label
  score = k_means.score(test)
  #names = [get_country_from_code(str(code)) for code in list(df['name'])]
  #df['full_name'] = names
  df['cluster'] = labels
  df.to_csv(outpath)
  return score

# Adds the kmeans-cluster to the dataset
def kmeans(inpath="data3.csv", outpath="data4.csv", covariancepath="cov.txt", n=6, m=10):
  k_means = cluster.KMeans(n_clusters=n)
  df = pd.read_csv(inpath)
  test = pd.read_csv(inpath)
  labels = list(df.columns)

  #Delte the non numerical entries for data manipulation
  del test[labels[0]]
  del df[labels[0]]
  del test['name']

  #Regularize
  for col in list(test.columns):
    bound = np.mean(list(test[col]))
    regs = [elt/bound for elt in list(test[col])]
    test[col] = regs
    df[col] = regs
  trans = np.transpose(test)
  trans.to_csv('transpose.csv')

  #Fit the k means to our test data
  k_means.fit(test)
  labels = list(k_means.labels_)  #The label
  score = k_means.score(test)
  names = [get_country_from_code(str(code)) for code in list(df['name'])]
  df['full_name'] = names
  df['cluster'] = labels
  df.to_csv(outpath)
  return score


def score_opt(max, path="data3.csv", error=80):
  prev = 0
  for i in range(2,max):
    new_score = kmeans(inpath=path,n=i)
    print abs(new_score - prev)
    if abs(new_score - prev) < error:
      print i
      print "Finished k means analysis"
      return kmeans(n=i)
    prev = new_score

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

def trans_pca(n=2, inpath="trans_nonames.csv", clusterpath="feature_cluster.csv", outpath="pca.csv"):
  df = pd.read_csv(inpath)
  clusters_df = pd.read_csv(clusterpath)
  labels = list(df.columns)
  names = list(df[labels[0]])
  del df[labels[0]]
  pca = PCA(n_components=n)
  df = pca.fit_transform(df)
  df = pd.DataFrame(df)
  df['name'] = names
  df['clusters'] = clusters_df['cluster']
  df.to_csv(outpath, index=False)

def get_transpose(inpath="full_data_reg.csv", outpath="trans_nonames.csv"):
  df = pd.read_csv(inpath)
  labels = list(df.columns)
  del df['name']
  new_df = df.transpose()
  new_df.to_csv(outpath, index=True)
  return new_df

# Adds the kmeans-cluster to the dataset
def feature_cluster(inpath="trans_nonames.csv", outpath="feature_cluster.csv", n=7):
  k_means = cluster.KMeans(n_clusters=n)
  df = pd.read_csv(inpath)
  labels = list(df.columns)

  #Delete the feature names
  del df[labels[0]]

  #Fit the k means to our test data
  k_means.fit(df)
  labels = list(k_means.labels_)  #The labels
  df['cluster'] = labels
  df.to_csv(outpath, index=False)
  return df.as_matrix()

def draw_clusters(inpath="pca.csv"):
  colors = ["red", "blue", "green", "orange", "purple", "gray", "white", "yellow", "violet", "gray"]
  df = pd.read_csv(inpath)
  x = df['0']
  y = df['1']
  colors = [colors[int(i)] for i in df['clusters']]
  plt.scatter(x, y, c=colors, alpha=0.5)
  plt.savefig('clusters.png')
  plt.show()

def indic_names(inpath="pca.csv", outpath="named_feature_clusters.csv"):
  df = pd.read_csv(inpath)
  indics = list(df['name'])
  full_names = [get_name_of_indicator(indic) for indic in indics]
  df['indicator_name'] = full_names
  df.to_csv(outpath, index=False)
  print "Done"

def extract_relevant_features(inpath="named_feature_clusters.csv", outpath="relevant.csv"):
  df = pd.read_csv(inpath)
  clusters = list(df['clusters'])
  names = list(df['name'])
  print len(names)
  cats = [0]*7
  #print clusters
  for j in range(0, len(clusters)):
    clusterno = cats[int(clusters[j])]
    if clusterno == 0:
      cats[int(clusters[j])] = [names[j]]
    else:
      cats[int(clusters[j])].append(names[j])
  sum = 0
  for group in cats:
    print len(group)
    sum += len(group)
  print sum
      #if clusters[]
      #clusters[i] = clusters[clusters[j]]

#regularize(inpath="full_data_clean.csv", outpath="full_data_reg.csv")
#get_transpose()
## Delete top row of input
feature_cluster(n=10)
trans_pca(n=20)
draw_clusters()
indic_names()

#extract_relevant_features()