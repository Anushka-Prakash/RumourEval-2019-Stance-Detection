
'''This notebook file trains a MLP on TF-IDF features to predict
the stance of a text given its target. This model is trained and
saved. It is loaded while training with RoBERTa. The weights 
learned from this model are concatenated with RoBERTa's pooled output.

This code was partly adapted from - https://github.com/donglinchen/text_classification/blob/master/model_pytorch.ipynb'''

#Importing necessary libraries
import re
import scipy
import pandas         as pd
import io
import numpy          as np
import copy

import torch

from sklearn.metrics                  import classification_report
from sklearn.feature_extraction.text  import TfidfVectorizer

from torch                            import nn, optim
from torch.utils                      import data
from processing_data                  import dataDf


HIDDEN_LAYER_UNITS = 128
CLASS_NAMES        = ['support', 'deny', 'query', 'comment']
EPOCHS             = 55

trainDf = dataDf[0]
devDf   = dataDf[1]
testDf  = dataDf[2]

x_train = trainDf['TextSrcInre'].tolist()
y_train = trainDf['labelValue'].tolist()

x_dev  = devDf['TextSrcInre'].tolist()
y_dev  = devDf['labelValue'].tolist()

x_test = testDf['TextSrcInre'].tolist()
y_test = testDf['labelValue'].tolist()

#Instantiating TfidfVectorizer object and fitting it on the training set
tfidf         = TfidfVectorizer(min_df = 10, max_df = 0.5, ngram_range=(1,2))
x_train_feats = tfidf.fit(x_train)

#print(x_train_feats)
#print(len(x_train_feats.get_feature_names()))

x_train_transform = x_train_feats.transform(x_train)
#Converting the TF-IDF matrix to tensor
tfidf_transform_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(x_train_transform)).float()
print(x_train_transform.shape)

#Tranforming the development and test data to tf-idf matrix
x_dev  = tfidf.transform(x_dev)
x_test = tfidf.transform(x_test)

x_dev  = torch.tensor(scipy.sparse.csr_matrix.todense(x_dev)).float()
x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()

#Converting prections for train, dev and test data to tensors
y_train = torch.tensor(y_train)
y_dev   = torch.tensor(y_dev)
y_test  = torch.tensor(y_test)

class Tfidf_Nn(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden  = nn.Linear(len(tfidf.get_feature_names()), HIDDEN_LAYER_UNITS)
        # Output layer
        self.output  =  nn.Linear(HIDDEN_LAYER_UNITS, len(CLASS_NAMES))
        self.dropout = nn.Dropout(0.1)
        
        # Defining tanh activation and softmax output 
        self.tanh    = nn.Tanh()                                     #Using tanh as it performed better than ReLu during hyper-param optimisation
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of the below operations
        x = self.hidden(x)
        #print(x.shape)
        y = self.tanh(x)
        #print(y.shape)
        z = self.dropout(y)
        #print(z.shape)
        z = self.output(z)
        #print(z.shape)
        z = self.softmax(z)
        
        #returning the output from hidden layer and the output layer
        return  y, z


model = Tfidf_Nn()

'''Using class-weights to accomodate heavily imbalanced data. 
These weights were learnt by running several experiments using 
other weights and the weights that produced the best results have
 finally been used here'''

weights       = [8.0, 20.0, 8.0, 1.0]
class_weights = torch.FloatTensor(weights)
criterion     = nn.CrossEntropyLoss(weight = class_weights)

# Forward pass, get our logits
hidden_state_output, classfier_output = model(tfidf_transform_tensor)
print(classfier_output)
print(classfier_output[0].shape)

loss = criterion(classfier_output, y_train)

loss.backward()

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.Adam(model.parameters(), lr=0.02)

def train_model():
  train_losses = []
  dev_losses = []
  dev_accuracies = []
 
  for e in range(EPOCHS):
    
    correct_predictions = 0
    optimizer.zero_grad()
    hidden_layer_output, classifier_output = model.forward(tfidf_transform_tensor)

    loss = criterion(classifier_output, y_train)
    loss.backward()
    train_loss = loss.item()
    train_losses.append(train_loss)
     
    optimizer.step()
    with torch.no_grad():
      model.eval()
 
      #Getting hidden layer and softmax output from model for dev data
      hidden_layer_output, classifier_output = model(x_dev)
         
      dev_loss = criterion(classifier_output, y_dev)
      dev_losses.append(dev_loss)
 
      _, preds = torch.max(classifier_output, dim=1)
      correct_predictions += torch.sum(preds == y_dev)
         
      dev_accuracy = correct_predictions.double() / len(y_dev)
      dev_accuracies.append(dev_accuracy)
 
    model.train()
 
    print(f"Epoch: {e+1}/{EPOCHS}.. ",
           f"Training Loss: {dev_loss:.3f}.. ",
           f"Dev Loss: {dev_loss:.3f}.. ",
           f"Dev Accuracy: {dev_accuracy:.3f}")

train_model()

'''This function gets the predictions for each data point 
in the deevelopment and the training set'''

def get_predictions(model, x_test, y_test):

  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    model.eval()
    labels = y_test

    #Currently, not interested in the hidden layer outputs.
    _,classifier_output = model(x_test)

    #Not interested in the maximum values, interested with the indices of these max values
    _, preds = torch.max(classifier_output, dim=1)

    predictions.extend(preds)
    prediction_probs.extend(classifier_output)
    real_values.extend(labels)
  predictions = torch.stack(predictions)

  prediction_probs = torch.stack(prediction_probs)
  real_values = torch.stack(real_values)
  return  predictions, prediction_probs, real_values

#Getting predictions for the development set
y_pred_dev, y_pred_probs, y_true_dev = get_predictions(
  model,
  x_dev, 
  y_dev
)

#Getting the predictions for the test set
y_pred_test, y_pred_probs, y_true_test = get_predictions(
  model,
  x_test, 
  y_test
)


