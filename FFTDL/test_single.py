"""
 used for only image model
"""
import hydra
import hydra.utils
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
import matplotlib.pyplot as plt

from utils import MPIDataset,save_checkpoint, load_checkpoint,Evaluator
from models.Classifiers import *

import random 
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1997)

@hydra.main(config_path="config",config_name="test_single.yaml")
def test_model(cfg:DictConfig):
    classifiers_list=list(cfg.classifiers.keys())

    model_pred_list=[]
    ##Loop over all the models
    for i in classifiers_list:
        model_config=cfg.classifiers[i]
        print(model_config)
        Classifier=eval(model_config.type)
        classifier=Classifier().to(device)  ## wait to ask
        
        ## load the checkpoint, checkpoint is a dict which has epoch, loss, model......
        ## https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        checkpoint_dir=Path(hydra.utils.to_absolute_path(model_config.checkpoint_dir))
        load_path=Path(checkpoint_dir)/model_config.checkpoint_name
        checkpoint=torch.load(load_path)
        classifier.load_state_dict(checkpoint["model"])  
        
        #### Step 1
        print(Path(model_config.dataset_dir)/'test_set.csv')
        MPI_testset=MPIDataset(Path(model_config.dataset_dir)/'test_set.csv')    
        MPI_dataloader=DataLoader(
            MPI_testset,
            batch_size=32,
            shuffle=False,
            #num_workers=0,   ## if had
            #pin_memory=True,
            drop_last=False
        )
        
        ## test begin
        cnn_evaluator=Evaluator()
        statistics,tabular_report=cnn_evaluator.evaluate(classifier,MPI_dataloader)
        print('MPI test:')
        print(statistics)
#         print('average accuracy is : {}'.format(np.mean(statistics['average_acc'])))
#         print('average precision is : {}'.format(np.mean(statistics['average_precision'])))
#         print('recall is : {}'.format(np.mean(statistics['recall'])))
#         print('Area Under the Receiver Operating Characteristic Curve (ROC AUC) is : {}'.format(np.mean(statistics['auc'])))
        print(tabular_report)
        disp=metrics.ConfusionMatrixDisplay(statistics['confusion_matrix']) ## display labels not configurate
        disp.plot()
        



## confusion matrix and confusion matrix display!

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]
        
if __name__ == "__main__":
    test_model()