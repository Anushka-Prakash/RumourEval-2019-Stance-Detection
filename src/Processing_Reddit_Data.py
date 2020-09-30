'''This file processes the Reddit dataset available for RumuorEval 2019 into dataframes and stores them into CSV files.
The data from the CSV files is used for training and testing the models.'''

#Importing necessary libraries
import os
import pandas as pd
import json
import numpy  as np
from DataPaths import path_train_key,path_dev_key, path_test_key, training_data_path, test_data_path, reddit_train_data_path,reddit_dev_data_path, reddit_test_data_path

#Specifying the path to access Twitter key files which contains Twitter post IDs and their corresponding labels
train_key_df = pd.read_json(path_train_key)
dev_key_df = pd.read_json(path_dev_key)
test_key_df = pd.read_json(path_test_key)

#Processing the key file to obtain only Reddit data - Train , Development and Test
def processRedditKeyDataFrame(key_df, datasetType):
    key_taska_df = pd.DataFrame(key_df['subtaskaenglish'].dropna())
    
    #Resetting index and column names for reddit key data frames. 
    key_taska_df = key_taska_df.reset_index()
    key_taska_df = key_taska_df.rename(columns={'index': 'id', 'subtaskaenglish': 'label'})
    
    if datasetType   ==  'train':
        reddit_key_tasks_df = key_taska_df[4519:] #1st 4519 records are keys for Twitter posts
    elif datasetType ==  'dev':
        reddit_key_tasks_df = key_taska_df[1049:] #1st 1049 records are keys for Twitter posts 
    elif datasetType ==  'test':
        reddit_key_tasks_df =  key_taska_df[1066:] #1st 1066 records are keys for Twitter posts
    return reddit_key_tasks_df
        

#Loading the key for Reddit Train, development and test data
reddit_train_key_df = processRedditKeyDataFrame(train_key_df, 'train')
reddit_dev_key_df   = processRedditKeyDataFrame(dev_key_df, 'dev')
reddit_test_key_df  = processRedditKeyDataFrame(test_key_df, 'test')

'''This function fetches the source post from each tree based conversation thread (consisting of source and replies
on the source post). Data from each source post is stored in a dictionary containing information about source text, 
source ID, and the ID of the post that this post is replying to. This information about all training exmaples is 
stored in a list'''
def processRedditSourcePosts(reddit_dataset_path):
    reddit_dirs = next(os.walk(reddit_dataset_path))[1] #Accessing Reddit directories
    reddit_dirs_sorted = sorted(reddit_dirs)
    
    reddit_src_dirs  = []
    reddit_src_posts = []
    
    for directory in reddit_dirs_sorted:
        reddit_src_path = reddit_dataset_path + '/' + directory + '/source-tweet' #accessing source directories
        reddit_src_dirs.append(next(os.walk(reddit_src_path))[2])
    
    src_reddit_files = []
    for sdirs in reddit_src_dirs:
        for i in sdirs:
            src_reddit_files.append(i)
    src_reddit_files_sorted = sorted(src_reddit_files)
    
    for file in src_reddit_files_sorted:
        paths = reddit_dataset_path + '/' + file.split('.')[0] + '/source-tweet' + '/' + file #accessing source files
        reddit_post_dict = {}
        
        #Loading data from each Reddit JSON file containing information about the source post
        with open(paths) as f:
            for line in f:
                src = json.loads(line)
                text = src['data']['children'][0]['data']['title']
                rid = src['data']['children'][0]['data']['id']
                
                reddit_post_dict['text'] = text  #Loading source text
                reddit_post_dict['id'] = rid     #Loading source ID
                
                #Loading in-reply ID which is the ID of the post this post has replied to.
                #Since we are loading the source post here , this post is not actually replying to any post. 
                reddit_post_dict['inre'] = 'None'
                reddit_src_posts.append(reddit_post_dict)
                
    #print(twitter_src_posts) 
    #print(len(twitter_src_posts))
    return reddit_dirs_sorted, reddit_src_posts


'''Converting source post list for training, dev and test data (Each element of the list is a dictionary containning source text, source post ID and the in-reply ID)
to a Dataframe'''
reddit_train_dirs_sorted , reddit_train_src_posts = processRedditSourcePosts(reddit_train_data_path)
reddit_train_src_posts_df = pd.DataFrame(reddit_train_src_posts)

reddit_dev_dirs_sorted , reddit_dev_src_posts = processRedditSourcePosts(reddit_dev_data_path)
reddit_dev_src_posts_df = pd.DataFrame(reddit_dev_src_posts)

reddit_test_dirs_sorted , reddit_test_src_posts = processRedditSourcePosts(reddit_test_data_path)
reddit_test_src_posts_df = pd.DataFrame(reddit_test_src_posts)


'''This function fetches the reply post from each tree based conversation thread (consisting of source and replies
on the source post). Data from each reply post is stored in a dictionary containing information about reply text, 
reply ID, and the ID of the post that this post is replying to. This information about all training exmaples is 
stored in a list'''
def processRedditReplyPosts(reddit_dataset_path, reddit_dirs_sorted):
    replies_files = []
    reddit_replies = []

    for directory in reddit_dirs_sorted:
        reddit_src_path = reddit_dataset_path + '/' + directory + '/replies' #Accessing the replies directory
        replies_files.append(next(os.walk(reddit_src_path))[2])
        
        for i in (next(os.walk(reddit_src_path))[2]):
            paths = reddit_dataset_path + '/' + directory + '/replies' + '/' + i #Accesing each reply file
            reddit_post_dict = {}
            with open(paths) as f:
                for line in f:
                    src = json.loads(line)
                    rid = src['data']['id']
                    inre = src['data']['parent_id']
                    
                    '''A few replies do not have any text data. This was because some of the replies were 
                    deleted but they were kept as is in the rumourEval data'''
                    
                    if 'body' in src['data']: 
                        text = src['data']['body']

                    reddit_post_dict['text'] = text               #Loading reply text
                    reddit_post_dict['id'] = rid                  #Loading ID of the reply post
                    reddit_post_dict['inre'] = inre.split('_')[1] #Loading ID of the post that this post replied to
                    reddit_post_dict['source'] = directory        #Laoding the ID of the source post of this conversation thread
                    reddit_replies.append(reddit_post_dict)
                    
   
    #print(twitter_replies) 
    #print(len(twitter_replies))
    return reddit_replies


'''Converting reply post list for train, dev and test data (Each element of the list is a dictionary containing reply text, reply post ID, the in-reply ID and the ID 
of the source post of that conversation thread) to a Dataframe'''
reddit_train_replies    = processRedditReplyPosts(reddit_train_data_path, reddit_train_dirs_sorted)
reddit_train_replies_df = pd.DataFrame(reddit_train_replies)

reddit_dev_replies    = processRedditReplyPosts(reddit_dev_data_path, reddit_dev_dirs_sorted)
reddit_dev_replies_df = pd.DataFrame(reddit_dev_replies)

reddit_test_replies    = processRedditReplyPosts(reddit_test_data_path, reddit_test_dirs_sorted)
reddit_test_replies_df = pd.DataFrame(reddit_test_replies)

'''This function concatenates the reply dataframe with the source dataframe to form 1 dataframe. 
It further removes any white spaces that could be present with data in id and in-reply ID columns. 
These columns will later be used to join dataframes'''

def redditCleanDf(src_posts_df, replies_df):
    reddit_data = [src_posts_df, replies_df]

    reddit_data = pd.concat(reddit_data)

    reddit_data['id'] = reddit_data.id.astype(str)
    reddit_data['inre'] = reddit_data.inre.astype(str)
    
    reddit_clean_data = pd.DataFrame(reddit_data)
    
    reddit_clean_data.id = reddit_clean_data.id.str.strip()     #Strips out any white spaces in the id column
    reddit_clean_data.inre = reddit_clean_data.inre.str.strip() #Strips out any white spaces in the inre column
    return reddit_clean_data


#Loading the clean reddit dataframe containing data from training, dev and test sets
reddit_clean_train_df = redditCleanDf(reddit_train_src_posts_df, reddit_train_replies_df)
reddit_clean_dev_df   = redditCleanDf(reddit_dev_src_posts_df, reddit_dev_replies_df)
reddit_clean_test_df  = redditCleanDf(reddit_test_src_posts_df, reddit_test_replies_df)

#Merging the data (training, dev and test) from the post and the key dataframes to combine each post with its associated label
reddit_train_withKeys_df = pd.merge(reddit_clean_train_df, reddit_train_key_df, how = 'inner', on = "id", )
reddit_dev_withKeys_df   = pd.merge(reddit_clean_dev_df, reddit_dev_key_df, how = 'inner', on = "id", )
reddit_test_withKeys_df  = pd.merge(reddit_clean_test_df, reddit_test_key_df, how = 'inner', on = "id", )


'''This function accepts the dataframe containing all the necessary information related to 1 post along with their
labels as parameter. It then creates a 2 new dataframe that consists of each post and its ID renaming these as inre 
and inreText and source and sourceText respectively. These 2 dataframes are independently joined with the dataframe 
containin all the information to produce texts corresponding to reply IDs and source IDs'''

def fetchRedditDataset(reddit_withKeys_df):
    
    reddit_df = reddit_withKeys_df[['id', 'text']].copy()
    
    reddit_df_new = reddit_df.rename(columns={'id': 'inre', 'text': 'inreText'})
    reddit_df_new1 = reddit_df.rename(columns={'id': 'source', 'text': 'sourceText'})
    
    reddit_dataset = pd.merge(reddit_withKeys_df, reddit_df_new, how = 'left', on = "inre", )
    reddit_dataset1 = pd.merge(reddit_withKeys_df, reddit_df_new1, how = 'left', on = "source", )
    
    return reddit_dataset, reddit_dataset1

#Creating dataframe containing source post text and reply post text for reddit training, dev and test data
reddit_train_dataset_inre, reddit_train_dataset_src = fetchRedditDataset(reddit_train_withKeys_df)
reddit_dev_dataset_inre, reddit_dev_dataset_src= fetchRedditDataset(reddit_dev_withKeys_df)
reddit_test_dataset_inre, reddit_test_dataset_src = fetchRedditDataset(reddit_test_withKeys_df)

#Merging the 2 dataframes containig source text and reply text for training, dev and test data
reddit_train_dataset_src = pd.merge(reddit_train_dataset_inre, reddit_train_dataset_src, how = 'inner', on = "id",)
reddit_dev_dataset_src = pd.merge(reddit_dev_dataset_inre, reddit_dev_dataset_src, how = 'inner', on = "id",)
reddit_test_dataset_src = pd.merge(reddit_test_dataset_inre, reddit_test_dataset_src, how = 'inner', on = "id",)

#Removing unnecessary columns for reddit training, dev and test data
reddit_new_train_data_df = reddit_train_dataset_src[['text_x', 'id', 'inre_x', 'source_x' ,'label_x','inreText', 'sourceText' ]].copy()
reddit_new_dev_data_df = reddit_dev_dataset_src[['text_x', 'id', 'inre_x', 'source_x' ,'label_x','inreText', 'sourceText' ]].copy()
reddit_new_test_data_df = reddit_test_dataset_src[['text_x', 'id', 'inre_x', 'source_x' ,'label_x','inreText', 'sourceText' ]].copy()

'''If the reply is directly to a source post, then the in-reply post and the source post for this reply post will be the same.
Hence to avoid redundant data, this block replaces the source text with nan values for training data '''
def removeRedundantData(reddit_df):
    for i in range(0,len(reddit_df)):
        if reddit_df['inre_x'][i] == reddit_df['source_x'][i]:
            reddit_df['sourceText'][i] = np.nan
    return reddit_df
        
reddit_new_train_data_df = removeRedundantData(reddit_new_train_data_df)
reddit_new_dev_data_df   = removeRedundantData(reddit_new_dev_data_df)
reddit_new_test_data_df  = removeRedundantData(reddit_new_test_data_df)
    
'''Saving the final train, development and test data frames for Reddit data into CSVs. The data from these CSV files are further
used in the NLP models'''
reddit_new_train_data_df.to_csv('RedditTrainDataSrc.csv', encoding='utf-8', index=False)
reddit_new_dev_data_df.to_csv('RedditDevDataSrc.csv', encoding='utf-8', index=False)
reddit_new_test_data_df.to_csv('RedditTestDataSrc.csv', encoding='utf-8', index=False)
