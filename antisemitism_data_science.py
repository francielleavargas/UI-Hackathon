# -*- coding: utf-8 -*-
#/usr/local/lib/python3.6

# JSON data stats:
# Total entries in dictionary: 1005
# Total annotated entries: 1005
# Total entries classified as antisemitic: 436
# Total entries classified as not antisemitic: 569


from collections import Counter
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer 
import numpy as np
import pandas as pd
import string
from textblob import TextBlob
import langid
import emoji
import spacy
import json
import nltk
import re


#Variable declarations-------------------------------------------------------
nlp = spacy.load("en_core_web_sm")
nlp_1 = spacy.load("en_core_web_md")

#Functions-------------------------------------------------------------------
def hash_fix(h):
	h1 = re.sub(r'[0-9]+', '', h)
	h2 = re.sub(r'#', '', h1)
	h3 = segment(str(h2))
	h4 = ' '.join(map(str, h3)) 
	return h4

def remove_punct(text):
	text  = "".join([char for char in text if char not in string.punctuation])
	text = re.sub('[0-9]+', '', text)
	return text

def give_emoji_free_text(text):
	allchars = [str for str in text]
	emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
	clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
	return clean_text

def extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

def remove_repetidos(lista):
	l = []
	for i in lista:
		if i not in l:
			l.append(i)
	l.sort()
	return l

#Create a dataframe-------------------------------------------------------
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


#Classes balancing----------------------------------------------------------- 
classe_1  = tweets_df[tweets_df.antisemitism_rating == '1'].head(285)
classe_2 = tweets_df[tweets_df.antisemitism_rating == '2']
classe_3 = tweets_df[tweets_df.antisemitism_rating == '3']
classe_4 = tweets_df[tweets_df.antisemitism_rating == '4']
classe_5 = tweets_df[tweets_df.antisemitism_rating == '5']

#Slip in binary class
c0 = pd.concat([classe_1, classe_2, classe_3])
c1 = pd.concat([classe_4, classe_5])
c3 = pd.concat([c0, c1])

data = []
ner =[]
#Starting the dataset exploration and annotation------------------------------
for ad, texts in enumerate(c3['tweet']):
	
	#Data cleaning
	texts = texts.lower() 
	texts = texts.lstrip()
	texts = texts.replace(r"(http|@)\S+", "")
	texts = texts.replace(r"::", ": :")
	texts = texts.replace(r"’", "")
	texts = texts.replace(r"’", "")
	texts = texts.replace(r"|", "")
	texts = texts.replace(r"/", "")
	texts = texts.replace(r"", "")
	texts = texts.replace(r"'", "")
	texts = texts.replace(r"*", "")
	texts = texts.replace(r"!", "")
	texts = texts.replace(r"?", "")
	texts = texts.replace(r"«", "")
	texts = texts.replace(r"»", "")
	texts = texts.replace(r"(", "")
	texts = texts.replace(r"‘", "")
	texts = texts.replace(r"“", "")
	texts = texts.replace(r"”", "")
	texts = texts.replace(r"“", "")
	texts = texts.replace(r";", "")
	texts = texts.replace(r",", "")
	texts = texts.replace(r".", "")
	texts = texts.replace(r'"', "")
	texts = texts.replace(r':', "")
	texts = texts.replace(r'_', "")
	texts = texts.replace(r'&', "")
	texts = texts.replace(r")", "")
	texts = texts.replace(r'¿', "")
	texts = texts.replace(r"[^a-z\':_]", " ")
	texts = texts.replace(r"(can't|cannot)", 'can not')
	texts = texts.replace(r"n't", ' not')
	texts = re.sub('@[^\s]+','',texts) #remove usernames
	texts = give_emoji_free_text(texts)
	texts = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", texts).split())
	texts = re.sub(r'http\S+', '', texts)
		#texts = re.sub('[0-9]+', '', texts) #remove pontuaction
	#texts = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", texts)
	#c1['hashtag'] = c1['tweet'].apply(lambda x: re.findall(r'\B#\w*[a-zA-Z]+\w*', texts)) #creating a new column
	#texts = re.findall(r"#(\w+)", texts)
	
	doc = nlp(texts)
	#print(ad, doc)

	#Dependece Tree
	#for token in doc:
		#print(token.text, ';', token.dep_, ';', token.head.text, ';', token.head.pos_,)

	#POS tagging
	#for token in doc:
		#if token.tag == 'NOUN' or token.tag == 'VERB' or token.tag == 'ADJ' or token.tag == 'ADV': 
		#print(token.text, ';', token.lemma_, ';', token.pos_, ';', token.tag_, ';', token.dep_, ';', token.shape_, ';', token.is_alpha, ';', token.is_stop)
	
	#NER annotation
	for ent in doc.ents:
		if (ent.label_ == 'EVENT') or (ent.label_ == 'FAC') or (ent.label_ == 'GPE') or (ent.label_ == 'LOC') or (ent.label_ == 'NORP') or (ent.label_ == 'ORG') or (ent.label_ == 'PERSON'):
			#print(ad, ent.text, ent.label_)
			lang = langid.classify(ent.text)
			if 'en' in lang:
				lemmatizer = WordNetLemmatizer()
				ner.append(lemmatizer.lemmatize(ent.text))

	ner = remove_repetidos(ner)
	lista = [0 for j in range(len(ner))]
	

					
