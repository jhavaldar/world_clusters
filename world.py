import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy
from sklearn import cluster
from sklearn.feature_selection import VarianceThreshold

codes_dict = {}

def print_dict(input, sep=' : '):
  for key in input:
    print str(key) + sep + str(input[key])

def get_soup(url):
  r = requests.get(url)
  page_text = r.text.encode('utf-8').decode('ascii', 'ignore').replace("wb:", '')
  soup = BeautifulSoup(page_text, 'lxml')
  return soup

#Returns the following JSON data: topic index, topic name
def get_topics():
  dict = {}
  url ="http://api.worldbank.org/topics/"
  soup = get_soup(url)
  out = soup.find_all('topic')
  for elt in out:
    dict[str(elt['id'])] = elt.value.get_text()
  return dict

# Given an array of indicators, find the useful ones
def get_useful_indicators(arr,minimum):
  new_arr = [elt for elt in arr if is_useful(elt,minimum)]
  return new_arr

#Get total number of entries for given indicator
def get_total(indicator):
  base_url ="http://api.worldbank.org/countries/all/indicators/"
  url = base_url + indicator + "?format=xml&MRV=1"
  soup = get_soup(url)
  data = soup.data
  if data is None:
    return -1
  total = int(soup.data['total'])
  return total

# Given an array of indicators, find the ones above a certain number of entries
def is_useful(indicator, minimum):
  base_url ="http://api.worldbank.org/countries/all/indicators/"
  url = base_url + indicator + "?format=xml&MRV=1"
  soup = get_soup(url)
  data = soup.data
  if data is None:
    return False
  total = int(soup.data['total'])
  return total > minimum

# Given an array of topics ids, returns all the associated indicators
def get_indicators_from_topics(arr):
  dict = {}
  base_url ="http://api.worldbank.org/topic/"
  for i in arr:
    url = base_url+str(i)+"/indicator"
    soup = get_soup(url)
    out = soup.find_all('indicator')
    for elt in out:
      name = elt.find('name').get_text()
      dict[str(elt['id'])] = name
  return dict

# Returns all the indicators
def get_all_indicators():
  dict = {}
  url = "http://api.worldbank.org/indicators/"
  soup = get_soup(url)
  out = soup.find_all('indicator')
  for elt in out:
    name = elt.find('name').get_text()
    dict[str(elt['id'])] = name
  return dict

# Pick the various indicators we're going to use
def pick_indicators():
  topics =[14, 16, 3, 5, 4, 8, 21]
  dict = (get_indicators_from_topics(topics))
  arr = [key for key in dict]
  print "got indicators"
  filtered = get_useful_indicators(arr,200)
  print "got filter"
  with open("out.csv", "w") as fp:
      for indicator in filtered:
        fp.write(indicator+",")

#Populates country codes
def populate_codes_dict(inpath="ids_to_names.csv"):
  with open(inpath, "r") as fp:
    for row in fp:
      splits = row.split(",")
      codes_dict[splits[0]] = splits[1].strip()

def get_country_from_code(code, inpath="ids_to_names.csv"):
  if code in codes_dict:
    return codes_dict[code]
  else:
    return "N/A"

#Given array of indicators, return dict of countires, along with values if they exist
def get_data_from_indicators(arr):
  base_url = "http://api.worldbank.org/countries/all/indicators/"
  dict = {}
  i = 0.0
  total = len(arr)
  for j in range(0, total):
    indic = arr[j]
    url = base_url+indic+"?format=xml&MRV=1&page=1"
    soup = get_soup(url)
    container = soup.find('data')
    if container is not None:
      # Find the number of pages of data we have for this indicator
      total_pages = int(container['pages'])
      # For each of the pages, get the country id and associated value
      for i in range(1,total_pages+1):
        url = base_url+indic+"?format=xml&MRV=1&page="+str(i)
        soup = get_soup(url)
        container = soup.find('data')
        countries = container.find_all('data')
        for country in countries:
          value = country.value.get_text()
          id = country.country['id']
          # Check that the value is nonempty
          if value!="":
            # Check if the country id is already key in the dictionary
            if id not in dict:
              dict[id] = {}
            values = dict[id] # Get dict of values
            values[indic] = value
            dict[id] = values
    progress = (j+1.0)/total
    print "Getting data: " + str(progress)
  return dict

def write_indicator_data(chosen, filepath="data.csv"):
  dict = get_data_from_indicators(chosen)
  with open(filepath, "w") as fp:
    # Write the headers
    fp.write(",")
    for indic in chosen:
      fp.write(indic+",")
    fp.write("\n")
    for country_id in dict:
      fp.write(country_id)
      for indic in chosen:
        value = -1
        if indic in dict[country_id]:
          value = dict[country_id][indic]
        fp.write(","+str(value))
      fp.write("\n")

def clean_cols(inpath='data.csv', outpath='data1.csv', clean_pct=0.65):
  #Only keep columns which have at least 'thresh' percentage of non-empty cells
  matrix = []
  arr = []
  header = []

  with open(inpath, "r") as fp:
    for row in fp:
      row = row.split(",")
      if row[0]!='':
        matrix.append(row)
      else:
        header = row
  num_rows = len(matrix)
  num_cols = len(matrix[0])

  for col in range(1,num_cols):
    clean_entries = len([row[col] for row in matrix if float(row[col])!=-1.0])
    # Calculates the number of entries in a given
    e_clean_pct = (clean_entries+0.0)/num_rows
    if e_clean_pct < clean_pct:
      arr.append(col)

  new_matrix = []

  new_header = []
  for col_index in range(0, num_cols):
    if col_index not in arr:
      new_header.append(header[col_index])

  new_matrix.append(new_header)

  for row in matrix:
    new_row = []
    for col_index in range(0, num_cols):
      if col_index not in arr:
        new_row.append(row[col_index])
    new_matrix.append(new_row)

  print str(len(new_matrix[1])) + " columns remain"

  # Write it to the new thing!
  with open(outpath, "w") as fp:
    for row in new_matrix:
      fp.write((",").join(row))
      fp.write("\n")

# Delete rows which have more than a certain percentage of unnown data
def clean_rows(inpath='data1.csv', outpath='data2.csv', clean_pct=0.65):
  matrix = []
  new_matrix = []
  header = []

  with open(inpath, "r") as fp:
    for row in fp:
      row = row.split(",")
      if row[0]!='':
        matrix.append(row)
      else:
        new_matrix.append(row)
  num_cols = len(matrix[0])

  # For each row, check the number of -1s
  for row in matrix:
    clean_cols = len([entry for entry in row[1:] if float(entry)!=-1.0])
    e_clean_pct = (clean_cols/(num_cols-1.0))
    if e_clean_pct > clean_pct:
      new_matrix.append(row)

    # Write it to the new thing!
  with open(outpath, "w") as fp:
    for row in new_matrix:
      if row[0] not in bad_rows:
        fp.write((",").join(row))
        fp.write("\n")

  print str(len(new_matrix)) + " rows remain"

bad_rows = ['XM','B8','XH', '8S','T5','XE', 'ZQ', 'OE','V4','ZT','XF','XP','XT','XO','1W','V3','Z4','XC','EU','XU','ZJ','T2','V2','XD','Z7','T4','4E','XL','7E','XJ','XG','S1','XN','S4','T6','ZG','ZF','V1','XI','F1','1A','XQ','T3','T7']

def fill_missing_values(inpath="data2.csv", outpath="data3.csv"):
  df = pd.read_csv("data2.csv")
  col_labels = list(df.columns.values)

  for col_label in col_labels[1:]:
    # Locate the column, and calculate the mean of the non-present entries
    column = list(df[col_label])
    filter = [entry for entry in column if float(entry)!= -1.0]
    mean = numpy.mean(filter)
    # Replace the missing entries with the mean
    df[col_label] = df[col_label].replace(-1.0, mean)
    column = list(df[col_label])
    filter = [entry for entry in column if float(entry)!= -1.0]
  df.to_csv(outpath)

# Adds the kmeans-cluster to the dataset
def kmeans(inpath="data3.csv", outpath="data4.csv",n=6, m=10):
  k_means = cluster.KMeans(n_clusters=n)
  df = pd.read_csv(inpath)
  test = pd.read_csv(inpath)
  del test['Unnamed: 0']
  del df['Unnamed: 0']
  del test['Unnamed: 0.1']
  k_means.fit(test)
  labels = list(k_means.labels_)
  score = k_means.score(test)
  names = [get_country_from_code(str(code)) for code in list(df['Unnamed: 0.1'])]
  df['name'] = names
  df['cluster'] = labels
  df.to_csv(outpath)
  return score


chosen = ''
with open("out.csv", "r") as fp:
  arr = fp.read().split(",")
  chosen = arr

populate_codes_dict()
print "Populated country code"
#write_indicator_data(chosen)
#print "Wrote indicator data to a file"
clean_cols(clean_pct=0.90)
print "Cleaned by column"
clean_rows(clean_pct=0.95)
print "Cleaned by row"
fill_missing_values()
print "Filled missing value"
for i in range(1,10):
  print kmeans(n=8)

print "Finished k means analysis"
