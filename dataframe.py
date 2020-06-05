import numpy as np
import pandas as pd
import json

# Open json file from same folder, read in data, and convert to a dictionary
json_dict = {}
with open('hackathon.json') as file:
    json_dict = json.load(file)

# Total entries in dictionary: 1006
# Total annotated entries: 895
# Total entries classified as antisemitic: 420
# Total entries classified as not antisemitic: 475

# Create a dictionary with only annotated entries (895 total)
data_dict = {}
i = 0
for entry in json_dict:
    if 'antisemitism_rating' in entry:
        data_dict[str(i)] = entry
        i = i + 1

# Lookup entry 0's text attribute
print(data_dict['0']['text'])
