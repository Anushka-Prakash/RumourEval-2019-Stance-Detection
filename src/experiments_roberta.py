'''This file was used to experiment the RoBERTa and various ensemble models with RoBERTa on the given dataset.'''

#Importing necessary libraries
import tensorflow     as tf
import torch

import re
import scipy
import pandas         as pd
import io
import numpy          as np
import copy
import seaborn        as sns

import transformers
from transformers                     import  RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

from sklearn.metrics                  import classification_report
from sklearn.feature_extraction.text  import TfidfVectorizer

from torch                            import nn, optim
from torch.utils                      import data
from processing_data                  import dataDf

#Seeding for deterministic results
RANDOM_SEED = 64
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
   torch.cuda.manual_seed(RANDOM_SEED)
   torch.cuda.manual_seed_all(RANDOM_SEED) # gpu vars
   torch.backends.cudnn.deterministic = True  #needed
   torch.backends.cudnn.benchmark = False

CLASS_NAMES = ['support', 'deny', 'query', 'comment']
MAX_LENGTH = 200
BATCH_SIZE = 4
EPOCHS = 6
HIDDEN_UNITS = 128

tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-large')  #Use roberta-large or roberta-base

# Getting GPU device name.
device_name = tf.test.gpu_device_name()

if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')


# If a GPU is available
if torch.cuda.is_available():    
    #set device to GPU   
    device = torch.device("cuda")
    print('We will use the GPU -', torch.cuda.get_device_name(0))

# If no GPU is available
else:
    print('No GPU available. Using the CPU instead.')
    device = torch.device("cpu")

trainDf = dataDf[0]
devDf   = dataDf[1]
testDf  = dataDf[2]

class StanceDataset(data.Dataset):

  def __init__(self, firstSeq, secondSeq, TextSrcInre, labelValue,  tokenizer, max_len):
    self.firstSeq    = firstSeq      #First input sequence that will be supplied to RoBERTa
    self.secondSeq   = secondSeq     #Second input sequence that will be supplied to RoBERTa
    self.TextSrcInre = TextSrcInre   #Concatenation of reply+ previous+ src text to get features from 1 training example
    self.labelValue  = labelValue    #label value for each training example in the dataset
    self.tokenizer   = tokenizer     #tokenizer that will be used to tokenize input sequences (Uses BERT-tokenizer here)
    self.max_len     = max_len       #Maximum length of the tokens from the input sequence that BERT needs to attend to

  def __len__(self):
    return len(self.labelValue)

  def __getitem__(self, item):
    firstSeq    = str(self.firstSeq[item])
    secondSeq   = str(self.secondSeq[item])
    TextSrcInre = str(self.TextSrcInre[item])
    
    #Encoding the first and the second sequence to a form accepted by RoBERTa
    #RoBERTa does not use token_type_ids to distinguish the first sequence from the second sequnece.
    encoding = tokenizer.encode_plus(
        firstSeq,
        secondSeq,
        max_length = self.max_len,
        add_special_tokens= True,
        truncation = True,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

    return {
        'firstSeq' : firstSeq,
        'secondSeq' : secondSeq,
        'TextSrcInre': TextSrcInre,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labelValue'  : torch.tensor(self.labelValue[item], dtype=torch.long)
    }


def createDataLoader(dataframe, tokenizer, max_len, batch_size):
  ds = StanceDataset(
      firstSeq    = dataframe.pReplyText.to_numpy(),
      secondSeq   = dataframe.pPreviousPlusSrcText.to_numpy(),
      TextSrcInre = dataframe.TextSrcInre.to_numpy(),
      labelValue  = dataframe.labelValue.to_numpy(),
      tokenizer   = tokenizer,
      max_len     = max_len
  )

  return data.DataLoader(
      ds,
      batch_size  = batch_size,
      shuffle     = True,
      num_workers = 4
  )

#Creating data loader for training, dev and test data
trainDataLoader        = createDataLoader(trainDf, tokenizer, MAX_LENGTH, BATCH_SIZE)
developmentDataLoader  = createDataLoader(devDf, tokenizer, MAX_LENGTH, BATCH_SIZE)
testDataLoader         = createDataLoader(testDf, tokenizer, MAX_LENGTH, BATCH_SIZE)

tfidf = TfidfVectorizer(min_df = 10, max_df = 0.5, ngram_range=(1,2))

xtrain = trainDf['TextSrcInre'].tolist()
x_train_feats = tfidf.fit(xtrain)
print(x_train_feats)
print(len(x_train_feats.get_feature_names()))

x_train_transform = x_train_feats.transform(xtrain)
tfidf_transform_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(x_train_transform)).float()
print(x_train_transform.shape)

#This class defines the model that was used to train the MLP on TF-IDF features
class Tfidf_Nn(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(len(tfidf.get_feature_names()), HIDDEN_UNITS)
        # Output layer
        self.output =  nn.Linear(HIDDEN_UNITS, 4)
        self.dropout = nn.Dropout(0.1)
        
        # Defining tanh activation and softmax output 
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        #print(x.shape)
        y = self.tanh(x)
        #print(y.shape)
        z = self.dropout(y)
        #print(z.shape)
        z = self.output(z)
        #print(z.shape)
        z = self.softmax(z)
        
        #Returning the ouputs from the hidden layer and the final output layer
        return  y, z

#Loading the already trained MLP model.
#from google.colab import drive
#drive.mount('/content/gdrive')
mlpmodel = Tfidf_Nn()

#model_save_name = 'pre-trainedTfidf.pt'
#path = F"/content/gdrive/My Drive/{model_save_name}"

#mlpmodel.load_state_dict(torch.load(path))
#mlpmodel.eval()

'''This class defines the model that will be used for 
training and testing on the dataset.

Adapted from huggingFace
This RoBERTa model from huggingface outputs the last hidden states
and the pooled output by default. Pooled output is the classification 
token (1st token of the last hidden state) further processed by a Linear
layer and a Tanh activation function.

The pre-trained RoBERTa model is used as the primary model.
This class experiments with RoBERTa and its ensemble with TF-IDF features. 
roberta-only :            No ensembling. This just fine-tunes the RoBERTa model. 
                          The pooled output is passed through a linear layer and 
                          softmax function is finally used for preictions. 

roberta-tfIdf :           This model conatenates the 1st token of last-hidden layer
                          from RoBERTa with TF-IDF features. Various ways of this 
                          concatenation was experimented (using pooled output instead
                          of 1st token of last hidden layer etc)

roberta-preTrainedTfIdf : This model concatenates the pooled output from
                          RoBERTa with the hidden layer output from a pre-trained
                          SNN that was trained on TF-IDF features.

Used dropout to prevent over-fitting.'''

class StanceClassifier(nn.Module):

  def __init__(self,  n_classes):
    super(StanceClassifier, self).__init__()
    self.robertaModel              = RobertaModel.from_pretrained('roberta-large')    #use roberta-large or roberta-base
    self.model_TFIDF               = mlpmodel                                        #Pre-trained SNN trained with TF-IDF features

    self.drop                      = nn.Dropout(p = 0.3)

    self.output                    = nn.Linear(self.robertaModel.config.hidden_size, n_classes)

    self.input_size_tfidf_only     = self.robertaModel.config.hidden_size + len(tfidf.get_feature_names()) 
    self.dense                     = nn.Linear( self.input_size_tfidf_only,  self.input_size_tfidf_only)
    self.out_proj                  = nn.Linear( self.input_size_tfidf_only, n_classes)

    self.input_size_preTrain_tfidf = self.robertaModel.config.hidden_size +  4 #HIDDEN_UNITS
    self.out                       = nn.Linear(self.input_size_preTrain_tfidf, n_classes)
    
    self.softmax                   = nn.Softmax(dim = 1)

  def forward(self, input_ids, attention_mask, inputs_tfidf_feats, modelType):
    
    roberta_output     = self.robertaModel(
        input_ids      = input_ids,               #Input sequence tokens
        attention_mask = attention_mask )         #Mask to avoid performing attention on padding tokens
    #print(roberta_output[1].shape)

    if modelType   == 'roberta-only':
      pooled_output = roberta_output[1]           #Using pooled output
      output        = self.drop(pooled_output)
      output        = self.output(output)

    elif modelType == 'roberta-tfIdf':
    # soutput = roberta_output[1]---------        experimenting with pooled output 
      soutput = roberta_output[0][:, 0, :]        #taking <s> token (equivalent to [CLS] token in BERT)
      x       = torch.cat((soutput, inputs_tfidf_feats) , dim=1)
      x       = self.drop(x)
      x       = self.dense(x)
      x       = torch.tanh(x)
      output  = self.drop(x)
      output  = self.out_proj(x)

    elif modelType == 'roberta-preTrainedTfIdf':
      tfidf_hidddenLayer, tfidf_output = self.model_TFIDF(inputs_tfidf_feats)
      #print(tfidf_hidddenLayer.shape)
      #print(tfidf_output.shape)
    
      #Conactenating pooled output from RoBERTa with the hidden layer from the pre-trained SNN using TF-IDF features. 
      #pooled_output = torch.cat((roberta_output[1], tfidf_output) , dim=1)-------- Experimenting with Output of pre-trained SNN 
      pooled_output = torch.cat((roberta_output[1], tfidf_output) , dim=1)
      output        = self.drop(pooled_output)
      output        = self.out(output)
    
    return self.softmax(output)

#Instantiating a StanceClassifier object as our model and loading the model onto the GPU.
model = StanceClassifier(len(CLASS_NAMES))
model = model.to(device)
#print(model)

'''Using the same optimiser as used in BERT paper
with a different learning rate'''
optimizer = AdamW(model.parameters(), 
                  lr = 2e-6, 
                  correct_bias= False)

totalSteps = len(trainDataLoader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps = totalSteps
)

'''Using class-weights to accomodate heavily imbalanced data. 
These weights were learnt by running several experiments using 
other weights and the weights that produced the best results have
finally been used here'''

weights      = [8.0, 84.0, 8.0, 1.0]
classWeights = torch.FloatTensor(weights)
lossFunction = nn.CrossEntropyLoss(weight = classWeights).to(device)

#This function is used for training the model. 
def train_epoch(
  model,
  dataLoader,
  lossFunction,
  optimizer,
  device,
  scheduler,
  n_examples
):

  model = model.train()
  losses = []
  correctPredictions = 0

  for d in dataLoader:
    
    input_ids              = d["input_ids"].to(device)                           #Loading input ids to GPU
    attention_mask         = d["attention_mask"].to(device)                      #Loading attention mask to GPU
    labelValues            = d["labelValue"].to(device)                          #Loading label value to GPU
    textSrcInre            = d["TextSrcInre"]                                    
    tfidf_transform        = x_train_feats.transform(textSrcInre)
    tfidf_transform_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(tfidf_transform)).float().to(device)    #Loading TF-IDF features to GPU

    #Getting the output from our model (Object of StanceClassification class) for train data
    outputs = model(
      input_ids          = input_ids,
      attention_mask     = attention_mask,
      inputs_tfidf_feats = tfidf_transform_tensor,
      modelType          = 'roberta-only'
    )

    #Determining the model predictions
    _, predictionIndices = torch.max(outputs, dim=1)
    loss = lossFunction(outputs, labelValues)

    #Calculating the correct predictions for accuracy
    correctPredictions += torch.sum(predictionIndices == labelValues)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return np.mean(losses), correctPredictions.double() / n_examples

#This function is used for evaluating the model on the development and test set
def eval_model(
    model, 
    dataLoader, 
    lossFunction,
    device,
    n_examples
    ):
  
  model = model.eval()
  losses = []
  correctPredictions = 0

  with torch.no_grad():
    for d in dataLoader:

      input_ids              = d["input_ids"].to(device)                          #Loading input ids to GPU
      attention_mask         = d["attention_mask"].to(device)                     #Loading attention mask to GPU
      labelValues            = d["labelValue"].to(device)                         #Loading label values to GPU
      textSrcInre            = d["TextSrcInre"]
      tfidf_transform        = x_train_feats.transform(textSrcInre)
      tfidf_transform_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(tfidf_transform)).float().to(device)    #Loading TF-IDF features to GPU

      #Getting the softmax output from model for dev data
      outputs = model(
        input_ids          = input_ids,
        attention_mask     = attention_mask,
        inputs_tfidf_feats = tfidf_transform_tensor,
        modelType          = 'roberta-only'
      )

      #Determining the model predictions
      _, predictionIndices = torch.max(outputs, dim=1)
      loss = lossFunction(outputs, labelValues)

      #Calculating the correct predictions for accuracy
      correctPredictions += torch.sum(predictionIndices == labelValues)
      losses.append(loss.item())

  return np.mean(losses), correctPredictions.double() / n_examples

#fine tuning ROBERTa and validating it 

for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}')
  trainLoss, trainAccuracy = train_epoch(
    model,
    trainDataLoader,
    lossFunction,
    optimizer,
    device,
    scheduler,
    len(trainDf)
  )
  
  print(f'Training loss {trainLoss} Training accuracy {trainAccuracy}')

  devLoss, devAccuracy = eval_model(
    model,
    developmentDataLoader,
    lossFunction,
    device,
    len(devDf)
  )

  print(f'Development loss {devLoss} Development accuracy {devAccuracy}')
  print()
  
  print()

#This function gets the predictions from the model after it is trained.
def get_predictions(model, data_loader):

  model = model.eval()
  review_texta = []
  review_textb = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      textas                 = d["firstSeq"]
      textbs                 = d["secondSeq"]
      input_ids              = d["input_ids"].to(device)
      attention_mask         = d["attention_mask"].to(device)
      labels                 = d["labelValue"].to(device)
      textSrcInre            = d["TextSrcInre"]
      tfidf_transform        = tfidf.transform(textSrcInre)
      tfidf_transform_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(tfidf_transform)).float().to(device)

      #Getting the softmax output from model
      outputs = model(
        input_ids          = input_ids,
        attention_mask     = attention_mask,
        inputs_tfidf_feats = tfidf_transform_tensor,
        modelType          = 'roberta-only'
      )

      _, preds = torch.max(outputs, dim=1)     #Determining the model predictions

      review_texta.extend(textas)
      review_textb.extend(textbs)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(labels)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  
  return review_texta, review_textb, predictions, prediction_probs, real_values

#Getting model predictions on dev dataset
firstSeq_dev, secondSeq_dev, yHat_dev, predProbs_dev, yTest_dev = get_predictions(
  model,
  developmentDataLoader
)

#Printing classification report for dev dataset (Evaluating the model on Dev set)
print(classification_report(yTest_dev, yHat_dev, target_names= CLASS_NAMES))

#Saving the model onto the drive
#from google.colab import drive
#drive.mount('/content/gdrive')

#model_save_name = 'RoBERTaLarge_TFIDFV2.pt'
#path = F"/content/gdrive/My Drive/{model_save_name}" 
#torch.save(model.state_dict(), path)

#Getting model predictions on test dataset
firstSeq_test, secondSeq_test, yHat_test, predProbs_test, yTest_test = get_predictions(
  model,
  testDataLoader
)

#Printing classification report for test dataset (Evaluating the model on test set)
print(classification_report(yTest_test, yHat_test, target_names= CLASS_NAMES))

#Saving the predictions onto a CSV file for error analysis
zippedList =  list(zip(firstSeq_test, secondSeq_test, yHat_test, predProbs_test, yTest_test ))
dfObj = pd.DataFrame(zippedList, columns = ['Texta' , 'Textb', 'Ypred', 'YpredsProbs', 'label'])

#from google.colab import drive
#drive.mount('drive')

#dfObj.to_csv('dataPredsFromRoberta_TFIDFV2.csv')
#!cp dataPredsFromRoberta_TFIDFV2.csv "drive/My Drive/"
