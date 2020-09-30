'''This file processes the Twitter dataset available for RumuorEval 2019 into dataframes and stores them into CSV files.
The data from the CSV files is used for training and testing the models. '''

#Importing necessary libraries
import os
import pandas as pd
import json
import numpy as np
from DataPaths import path_train_key,path_dev_key, path_test_key, training_data_path, test_data_path, twitter_trainingDev_data_path, twitter_test_data_path

#Specifying the path to access Twitter key files which contains Twitter post IDs and their corresponding labels
train_key_df = pd.read_json(path_train_key)
dev_key_df = pd.read_json(path_dev_key)
test_key_df = pd.read_json(path_test_key)

#Processing the key file to obtain only Twitter data - Train , Development and Test
def processTwitterKeyDataFrame(key_df, datasetType):
    key_taska_df = pd.DataFrame(key_df['subtaskaenglish'].dropna())
    
    #Resetting index and column names for twitter key data frames. 
    key_taska_df = key_taska_df.reset_index()
    key_taska_df = key_taska_df.rename(columns={'index': 'id', 'subtaskaenglish': 'label'})
    
    if datasetType == 'train':
        twitter_key_tasks_df = key_taska_df[0:4519] #Records after 4519 are keys for Reddit posts
    elif datasetType == 'dev':
        twitter_key_tasks_df = key_taska_df[0:1049] #Records after 1049 are keys for Reddit posts
    elif datasetType == 'test':
        twitter_key_tasks_df =  key_taska_df[0:1066] #Records after 1066 are keys for Reddit posts
    return twitter_key_tasks_df

#Loading keys for Twitter training, dev and test data
twitter_train_key_df = processTwitterKeyDataFrame(train_key_df, 'train')
twitter_dev_key_df = processTwitterKeyDataFrame(dev_key_df, 'dev')
twitter_test_key_df = processTwitterKeyDataFrame(test_key_df, 'test')


'''This function fetches the source post from each tree based conversation thread (consisting of source and replies
on the source post). Data from each source post is stored in a dictionary containing information about source text, 
source ID, and the ID of the post that this post is replying to. This information about all training exmaples is 
stored in a list'''
def processTwitterSourcePosts(twitter_dataset_path):
    twitter_dirs = next(os.walk(twitter_dataset_path))[1]
    twitter_dirs_sorted = sorted(twitter_dirs)
    
    twitter_src_dirs = []
    twitter_src_posts = []
    
    for directory in twitter_dirs_sorted:
        tweet_src_path = twitter_dataset_path + '/' + directory + '/source-tweet' #accessing source directories
        twitter_src_dirs.append(next(os.walk(tweet_src_path))[2])
    
    src_tweet_files = []
    for sdirs in twitter_src_dirs:
        for i in sdirs:
            src_tweet_files.append(i)
    src_tweet_files_sorted = sorted(src_tweet_files)
    
    for file in src_tweet_files_sorted:
        paths = twitter_dataset_path + '/' + file.split('.')[0] + '/source-tweet' + '/' + file
        tweet_post_dict = {}
        
        #Loading data from each Twitter JSON file containing information about the source post
        with open(paths) as f:
            for line in f:
                src = json.loads(line)
                text = src['text']
                inre = src['in_reply_to_status_id']
                tid = src['id']
                
                tweet_post_dict['text'] = text #Loading source text
                tweet_post_dict['id'] = tid    #Loading source ID
                tweet_post_dict['inre'] = inre
                twitter_src_posts.append(tweet_post_dict)
    #print(twitter_src_posts) 
    #print(len(twitter_src_posts))
    return twitter_dirs_sorted, twitter_src_posts

'''Converting source post list for training, development and test data (Each element of the list is a dictionary containning source text, source post ID and the in-reply ID)
to a Dataframe'''
twitter_trainDev_dirs_sorted , twitter_trainDev_src_posts = processTwitterSourcePosts(twitter_trainingDev_data_path)
twitter_trainDev_src_posts_df = pd.DataFrame(twitter_trainDev_src_posts)

twitter_test_dirs_sorted , twitter_test_src_posts = processTwitterSourcePosts(twitter_test_data_path)
twitter_test_src_posts_df = pd.DataFrame(twitter_test_src_posts)


'''This function fetches the reply post from each tree based conversation thread (consisting of source and replies
on the source post). Data from each reply post is stored in a dictionary containing information about reply text, 
reply ID, and the ID of the post that this post is replying to. This information about all training exmaples is 
stored in a list'''
def processTwitterReplyPosts(twitter_dataset_path, twitter_dirs_sorted):
    replies_files = []
    twitter_replies = []


    for directory in twitter_dirs_sorted:
        tweet_src_path = twitter_dataset_path + '/' + directory + '/replies'
        replies_files.append(next(os.walk(tweet_src_path))[2])
        
        for i in (next(os.walk(tweet_src_path))[2]):
            paths = twitter_dataset_path + '/' + directory + '/replies' + '/' + i
            tweet_post_dict = {}
            with open(paths) as f:
                for line in f:
                    src = json.loads(line)
                    text = src['text']
                    inre = str(src['in_reply_to_status_id'])
                    tid = src['id']
                    tweet_post_dict['text'] = text          #Loading reply text
                    tweet_post_dict['id'] = tid             #Loading ID of the reply post
                    tweet_post_dict['inre'] = inre          #Loading ID of the post that this post replied to
                    tweet_post_dict['source'] = directory   #Laoding the ID of the source post of this conversation thread
                    twitter_replies.append(tweet_post_dict)
   
    #print(twitter_replies) 
    #print(len(twitter_replies))
    return twitter_replies


'''Converting reply post list for train, development and test data (Each element of the list is a dictionary containning reply text, reply post ID, the in-reply ID and the ID 
of the source post of that conversation thread) to a Dataframe'''
twitter_trainDev_replies    = processTwitterReplyPosts(twitter_trainingDev_data_path, twitter_trainDev_dirs_sorted)
twitter_trainDev_replies_df = pd.DataFrame(twitter_trainDev_replies)

twitter_test_replies     = processTwitterReplyPosts(twitter_test_data_path, twitter_test_dirs_sorted)
twitter_test_replies_df  = pd.DataFrame(twitter_test_replies)


'''This function concatenates the reply dataframe with the source dataframe to form 1 dataframe. 
It further removes any white spaces that could be present with data in id and in-reply ID columns. 
These columns will later be used to join dataframes'''
def twitterCleanDf(src_posts_df, replies_df):
    twitter_data = [src_posts_df, replies_df]

    twitter_data = pd.concat(twitter_data)

    twitter_data['id']   = twitter_data.id.astype(str)
    twitter_data['inre'] = twitter_data.inre.astype(str)
    #result.dtypes
    twitter_clean_data = pd.DataFrame(twitter_data)
    #df10
    twitter_clean_data.id   = twitter_clean_data.id.str.strip()       #Strips out any white spaces in the id column
    twitter_clean_data.inre = twitter_clean_data.inre.str.strip()   #Strips out any white spaces in the inre column
    return twitter_clean_data

#Loading the clean twitter dataframes for training, development and test data
twitter_clean_trainDev_df = twitterCleanDf(twitter_trainDev_src_posts_df, twitter_trainDev_replies_df)
twitter_clean_test_df     = twitterCleanDf(twitter_test_src_posts_df, twitter_test_replies_df)


#Merging the data (training, dev and test) from the post and the key dataframes to combine each post with its associated label
twitter_train_withKeys_df = pd.merge(twitter_clean_trainDev_df, twitter_train_key_df, how = 'inner', on = "id", )
twitter_dev_withKeys_df   = pd.merge(twitter_clean_trainDev_df, twitter_dev_key_df, how = 'inner', on = "id", )
twitter_test_withKeys_df  = pd.merge(twitter_clean_test_df, twitter_test_key_df, how = 'inner', on = "id", )


'''This function accepts the dataframe containing all the necessary information related to 1 post along with their
labels as parameter. It then creates a 2 new dataframe that consists of each post and its ID renaming these as inre 
and inreText and source and sourceText respectively. These 2 dataframes are independently joined with the dataframe 
containin all the information to produce texts corresponding to reply IDs and source IDs'''
def fetchTwitterDataset(twitter_withKeys_df):
    
    twitter_df       = twitter_withKeys_df[['id', 'text']].copy()
    twitter_df_new   = twitter_df.rename(columns={'id': 'inre', 'text': 'inreText'})
    twitter_df_new1  = twitter_df.rename(columns={'id': 'source', 'text': 'sourceText'})
    twitter_dataset  = pd.merge(twitter_withKeys_df, twitter_df_new, how = 'left', on = "inre", )
    twitter_dataset1 = pd.merge(twitter_withKeys_df, twitter_df_new1, how = 'left', on = "source", )
    
    return twitter_dataset, twitter_dataset1

#Creating dataframe containing source post text and reply post text for twitter training, dev and test data
twitter_train_dataset_inre, twitter_train_dataset_src = fetchTwitterDataset(twitter_train_withKeys_df)
twitter_dev_dataset_inre, twitter_dev_dataset_src= fetchTwitterDataset(twitter_dev_withKeys_df)
twitter_test_dataset_inre, twitter_test_dataset_src = fetchTwitterDataset(twitter_test_withKeys_df)

#Merging the 2 dataframes containig source text and reply text for training, dev and test data
twitter_train_dataset_src = pd.merge(twitter_train_dataset_inre, twitter_train_dataset_src, how = 'inner', on = "id",)
twitter_dev_dataset_src = pd.merge(twitter_dev_dataset_inre, twitter_dev_dataset_src, how = 'inner', on = "id",)
twitter_test_dataset_src = pd.merge(twitter_test_dataset_inre, twitter_test_dataset_src, how = 'inner', on = "id",)

#Removing unnecessary columns for twitter training, dev and test data
twitter_new_train_data_df = twitter_train_dataset_src[['text_x', 'id', 'inre_x', 'source_x' ,'label_x','inreText', 'sourceText' ]].copy()
twitter_new_dev_data_df = twitter_dev_dataset_src[['text_x', 'id', 'inre_x', 'source_x' ,'label_x','inreText', 'sourceText' ]].copy()
twitter_new_test_data_df = twitter_test_dataset_src[['text_x', 'id', 'inre_x', 'source_x' ,'label_x','inreText', 'sourceText' ]].copy()

'''If the reply is directly to a source post, then the in-reply post and the source post for this reply post will be the same.
Hence to avoid redundant data, this block replaces the source text with nan values for training data '''
def removeRedundantData(twitter_df):
    for i in range(0,len(twitter_df)):
        if twitter_df['inre_x'][i] == twitter_df['source_x'][i]:
            twitter_df['sourceText'][i] = np.nan
    return twitter_df

twitter_new_train_data_df = removeRedundantData(twitter_new_train_data_df)
twitter_new_dev_data_df   = removeRedundantData(twitter_new_dev_data_df)
twitter_new_test_data_df  = removeRedundantData(twitter_new_test_data_df)

'''Saving the final train, development and test data frames for Reddit data into CSVs. The data from these CSV files are further
used in the NLP models'''
twitter_new_train_data_df.to_csv('TwitterTrainDataSrc.csv', encoding='utf-8', index=False)
twitter_new_dev_data_df.to_csv('TwitterDevDataSrc.csv', encoding='utf-8', index=False)
twitter_new_test_data_df.to_csv('TwitterTestDataSrc.csv', encoding='utf-8', index=False)




