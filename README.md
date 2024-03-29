# RumourEval-2019-Stance-Detection

Incorporating Count-based features into pre-trained models for Improved Stance Detection

# Replicating Results
- The RumourEval 2019 data can be downloaded 
from [here](https://figshare.com/articles/RumourEval_2019_data/8845580).
- This data should be downloaded to the resources folder.
- The data can be pre-processed by running the DataPaths.py,  Processing_Twitter_Data.py and Processing_Reddit_Data.py files in the same order. 
 
# Colab

In addition to the pre-processing code in this repository, we also make available colab notebooks used for experimentation. 

- The model used for training the MLP on TF-IDF features can be found [here](https://colab.research.google.com/drive/1mQbK-nI0EWGymUFJJJQ_35nMnvNG4RaL?usp=sharing).
- All the experiemnts with RoBERTa model, including our proposed architecture can be found  [here](https://colab.research.google.com/drive/1eB8EMCwEE1_o5QOdC0gEejxqgkv6Q_cO?usp=sharing). 


# Citation

If you make use of this work, please cite:

```
@inproceedings{prakash2020incorporating,
  title={Incorporating Count-Based Features into Pre-Trained Models for Improved Stance Detection},
  author={Prakash, Anushka and Madabushi, Harish Tayyar},
  booktitle={Proceedings of the 3rd NLP4IF Workshop on NLP for Internet Freedom: Censorship, Disinformation, and Propaganda},
  pages={22--32},
  year={2020}
}  
```
