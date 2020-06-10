# -*- coding: utf-8 -*-
#/usr/local/lib/python3.6

# JSON data stats:
# Total entries in dictionary: 1005
# Total annotated entries: 1005
# Total entries classified as antisemitic: 436
# Total entries classified as not antisemitic: 569

import numpy as np
import pandas as pd
import json
import spacy
import re
from collections import Counter
nlp = spacy.load("en_core_web_sm")

# Open json file from same folder, read in data, and convert to a dictionary
json_dict = {}
with open('hackathon2.json') as file:
	json_dict = json.load(file)

# Create a list of tweets (text only)
tweets = []
for tweet in json_dict:
	# Some tweets are re-tweets, so need to grab full original tweet
	if 'retweeted_status' in tweet.keys():
		tweets.append(tweet['retweeted_status']['text'])
	else:
		tweets.append(tweet['text'])

# Create a dataframe using tweets list and json_dict
tweets_df = pd.DataFrame(tweets, columns=['tweet'])
tweets_df['antisemitism_rating'] = np.array([tweet['antisemitism_rating'] for tweet in json_dict])
tweets_df['still_exists'] = np.array([tweet['still_exists'] for tweet in json_dict])
tweets_df['in_english'] = np.array([tweet['in_english'] for tweet in json_dict])
tweets_df['sarcasm'] = np.array([tweet['sarcasm'] for tweet in json_dict])
tweets_df['additional_comments'] = np.array([tweet['additional_comments'] for tweet in json_dict])
tweets_df['disagree_with'] = np.array([tweet['disagree_with'] for tweet in json_dict])
tweets_df['sentiment_rating'] = np.array([tweet['sentiment_rating'] for tweet in json_dict])
tweets_df['calling_out'] = np.array([tweet['calling_out'] for tweet in json_dict])
tweets_df['is_about_the_holocaust'] = np.array([tweet['is_about_the_holocaust'] for tweet in json_dict])
tweets_df['ihra_section'] = np.array([tweet['ihra_section'] for tweet in json_dict])


#Classes balancing 
classe_1  = tweets_df[tweets_df.antisemitism_rating == '1'].head(285)
classe_2 = tweets_df[tweets_df.antisemitism_rating == '2']
classe_3 = tweets_df[tweets_df.antisemitism_rating == '3']
classe_4 = tweets_df[tweets_df.antisemitism_rating == '4']
classe_5 = tweets_df[tweets_df.antisemitism_rating == '5']

#Slip in binary class
class_A = pd.concat([classe_1, classe_2, classe_3])
class_N = pd.concat([classe_4, classe_5])

#Variable declarations
verb = []
noun = []
propn = []
adp = []
sym = []
aux = []
adj = []
punct = []
conj = []
pron = []
propn = []
adve = []
num = []
ax1 = []
ax2 = []
ax3 = []
ax4 = []

#Starting the dataset exploration and annotation
for ad, item in enumerate(class_A['tweet']):
	result = re.sub(r"http\S+", "", item)
	doc = nlp(result)
	
	#NER annotation
	for ent in doc.ents:
		print(ent.text, ';', ent.start_char, ';', ent.end_char, ';', ent.label_)
		if ent.label_ == 'PERSON':
			ax1.append(ent.text)
		if ent.label_ == 'LOC':
			ax2.append(ent.text)
		if ent.label_ == 'ORG':
			ax3.append(ent.text)
		if ent.label_ == 'MISC':
			ax4.append(ent.text)

print(len(ax1))
print(len(ax2))
print(len(ax3))
print(len(ax4))
