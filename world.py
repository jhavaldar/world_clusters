import requests
import re
import pandas as pd
import numpy
from sklearn import cluster
from sklearn.feature_selection import VarianceThreshold
from world_bank_api import get_country_from_code

# Adds the kmeans-cluster to the dataset
def kmeans(inpath="data3.csv", outpath="data4.csv",n=6, m=10):
  k_means = cluster.KMeans(n_clusters=n)
  df = pd.read_csv(inpath)
  test = pd.read_csv(inpath)
  labels = list(df.columns)

  #Delte the non numerical entries for data manipulation
  del test[labels[0]]
  del df[labels[0]]
  del test['country_id']

  #Fit the k means to our test data
  k_means.fit(test)
  labels = list(k_means.labels_)  #The label
  score = k_means.score(test)
  names = [get_country_from_code(str(code)) for code in list(df['country_id'])]
  df['name'] = names
  df['cluster'] = labels
  df.to_csv(outpath)
  return score


def score_opt(max, error=10**17):
  prev = 0
  for i in range(2,max):
    new_score = kmeans(n=i)
    print abs(new_score - prev)
    if abs(new_score - prev) < error:
      print i
      print "Finished k means analysis"
      return kmeans(n=i)
    prev = new_score

score_opt(100)

