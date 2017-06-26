from bs4 import BeautifulSoup
import pandas as pd
import numpy

# Helper function for printing a dictionary
def print_dict(input, sep=' : '):
  for key in input:
    print str(key) + sep + str(input[key])

# Returns a BeautifulSoup object associated with a given URL
def get_soup(url):
  r = requests.get(url)
  page_text = r.text.encode('utf-8').decode('ascii', 'ignore').replace("wb:", '')
  soup = BeautifulSoup(page_text, 'lxml')
  return soup

#Returns the following as JSON data: topic index, topic name
def get_all_topics():
  dict = {}
  url ="http://api.worldbank.org/topics/"
  soup = get_soup(url)
  out = soup.find_all('topic')
  for elt in out:
    dict[str(elt['id'])] = elt.value.get_text()
  return dict

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

# Given an array of indicators, find the indicator which has at least 'minimum' entries
def get_useful_indicators(arr,minimum):
  new_arr = [elt for elt in arr if has_min_entries(elt,minimum)]
  return new_arr

# Given an array of indicators, find the ones above a certain number of entries
def has_min_entries(indicator, minimum):
  base_url ="http://api.worldbank.org/countries/all/indicators/"
  url = base_url + indicator + "?format=xml&MRV=1"
  soup = get_soup(url)
  data = soup.data
  if data is None:
    return False
  total = int(soup.data['total'])
  return total > minimum

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

# Pick the various indicators from an array of topics which meet a minimum number of entries
def pick_indicators(minimum, indics=[14, 16, 3, 5, 4, 8, 21], outpath="out.csv"):
  dict = (get_indicators_from_topics(indics))
  arr = [key for key in dict]
  filtered = get_useful_indicators(arr,200)
  with open(outpath, "w") as fp:
      for indicator in filtered:
        fp.write(indicator+",")

#Returns dictionary of country codes and names
def get_codes_dict(inpath="ids_to_names.csv"):
  codes_dict = {}
  with open(inpath, "r") as fp:
    for row in fp:
      splits = row.split(",")
      codes_dict[splits[0]] = splits[1].strip()
  return codes_dict

# Given a certain country code, return the country_name; generates dictionary every time on the fly
def get_country_from_code(code, inpath="ids_to_names.csv"):
  codes_dict = get_codes_dict(inpath)
  if code in codes_dict:
    return codes_dict[code]
  else:
    return "N/A"

#Given array of indicators, return dict of countires, along with values for each indicator if they exist
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

# The dictionary of countries with indicator data is printed out to a file
def write_indicator_data(chosen, filepath="data.csv"):
  dict = get_data_from_indicators(chosen)
  with open(filepath, "w") as fp:
    # Write the headers
    fp.write("name,")
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

# Removes columns with lower than a certain percentage of unfilled entries
def clean_cols(inpath='data.csv', outpath='data1.csv', clean_pct=0.65):
  df = pd.read_csv(inpath)
  num_rows = df.shape[0]
  outdf = pd.read_csv(inpath)
  # For each column, check the percenteage of clean entries
  for col_index in list(df.columns)[1:]:
    column = list(df[col_index])
    clean_entries = len([entry for entry in column if float(entry)!=-1.0])
    e_clean_pct = (clean_entries+0.0)/num_rows
    if e_clean_pct < clean_pct:
      del outdf[col_index]
  print str(len(list(outdf.columns))) + " columns remain"
  outdf.to_csv(outpath)

bad_rows = ['XM','B8','XH', '8S','T5','XE', 'ZQ', 'OE','V4','ZT','XF','XP','XT','XO','1W','V3','Z4','XC','EU','XU','ZJ','T2','V2','XD','Z7','T4','4E','XL','7E','XJ','XG','S1','XN','S4','T6','ZG','ZF','V1','XI','F1','1A','XQ','T3','T7']

# Delete rows which have more than a certain percentage of unfilled_entries,
# or which are not countries (bad rows)
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
    clean_cols = len([entry for entry in row[2:] if float(entry)!=-1.0])
    e_clean_pct = (clean_cols/(num_cols-1.0))
    if e_clean_pct > clean_pct:
      new_matrix.append(row)

    # Write it to the new file
  with open(outpath, "w") as fp:
    for row in new_matrix:
      if row[1] not in bad_rows:
        fp.write((",").join(row))
        fp.write("\n")

  print str(len(new_matrix)) + " rows remain"

# Fill in missing labels
def fill_missing_values(inpath="data2.csv", outpath="data3.csv"):
  df = pd.read_csv("data2.csv")
  col_labels = list(df.columns.values)

  for col_label in col_labels[2:]:
    # Locate the column, and calculate the mean of the non-present entries
    column = list(df[col_label])
    filter = [entry for entry in column if float(entry)!= -1.0]
    mean = numpy.mean(filter)
    # Replace the missing entries with the mean
    df[col_label] = df[col_label].replace(-1.0, mean)
  del df[col_labels[0]]
  df.to_csv(outpath)
  print "Replaced missing values"

clean_cols(clean_pct=0.85)
clean_rows(clean_pct=0.85)
fill_missing_values()