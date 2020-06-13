import numpy as np
import pandas as pd
import json

# Open json file from same folder, read in data, and convert to a dictionary
json_dict = {}
with open('group4pretty.json') as file:
    json_dict = json.load(file)

# Create a list of tweets (text only)
tweets = []
for tweet in json_dict:
    # Some tweets are re-tweets, so need to grab full original tweet
    if 'retweeted_status' in tweet.keys():
        if 'otherfields' in tweet['retweeted_status'].keys() and 'extended_tweet' in tweet['retweeted_status']['otherfields'].keys():
            full_text = tweet['retweeted_status']['otherfields']['extended_tweet']
            tweets.append(full_text[11:full_text.find(',display_text_range:[')])
        else:
            tweets.append(tweet['retweeted_status']['text'])
    else:
        if 'otherfields' in tweet.keys() and 'extended_tweet' in tweet['otherfields'].keys():
            full_text = tweet['otherfields']['extended_tweet']
            tweets.append(full_text[11:full_text.find(',display_text_range:[')])
        else:
            tweets.append(tweet['text'])

# Create a dataframe using tweets list and json_dict
tweets_df = pd.DataFrame(tweets, columns=['tweet'])
tweets_df['still_exists'] = np.array([tweet['still_exists'] for tweet in json_dict])
tweets_df['in_english'] = np.array([tweet['in_english'] for tweet in json_dict])
tweets_df['sarcasm'] = np.array([tweet['sarcasm'] for tweet in json_dict])
tweets_df['additional_comments'] = np.array([tweet['additional_comments'] for tweet in json_dict])
tweets_df['antisemitism_rating'] = np.array([tweet['antisemitism_rating'] for tweet in json_dict])
tweets_df['disagree_with'] = np.array([tweet['disagree_with'] for tweet in json_dict])
tweets_df['sentiment_rating'] = np.array([tweet['sentiment_rating'] for tweet in json_dict])
tweets_df['calling_out'] = np.array([tweet['calling_out'] for tweet in json_dict])
tweets_df['is_about_the_holocaust'] = np.array([tweet['is_about_the_holocaust'] for tweet in json_dict])
tweets_df['ihra_section'] = np.array([tweet['ihra_section'] for tweet in json_dict])

tweets_df.to_csv('group4.csv')
