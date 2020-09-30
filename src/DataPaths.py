import pandas as pd
import os

training_data_path = os.path.abspath('../resources/rumourEval2019/rumoureval-2019-training-data')
test_data_path     = os.path.abspath('../resources/rumourEval2019/rumoureval-2019-test-data')

twitter_trainingDev_data_path = training_data_path + '/train-data/Twitter_data'
twitter_test_data_path        = test_data_path + '/twitter-test-data'

path_train_key = '../resources/rumourEval2019/rumoureval-2019-training-data/train-key.json'
path_dev_key   = '../resources/rumourEval2019/rumoureval-2019-training-data/dev-key.json'
path_test_key  = '../resources/rumourEval2019/final-eval-key.json'

reddit_train_data_path  =  training_data_path + '/reddit-training-data'
reddit_dev_data_path    =  training_data_path + '/reddit-dev-data'
reddit_test_data_path   =  test_data_path     + '/reddit-test-data'




