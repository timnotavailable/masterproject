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
import random 
import numpy as np
import pandas as pd

from utils import MPIDataset,save_checkpoint, load_checkpoint
from models.Classifiers import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1997)


@hydra.main(config_path="config",config_name="model_ensemble.yaml")
def ensemble_model(cfg:DictConfig):
    classifiers_list=list(cfg.classifiers.keys())
    label_sum=[]   ## predicted label
    num_classifier=len(classifiers_list)
    label=[]   ## true label
    

    for i in classifiers_list:
        model_config=cfg.classifiers[i]
        print(model_config)
        Classifier=eval(model_config.type)
        classifier=Classifier().to(device)        
        ## load the checkpoint, checkpoint is a dict which has epoch, loss, model......
        ## https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        checkpoint_dir=Path(hydra.utils.to_absolute_path(model_config.checkpoint_dir))
        load_path=Path(checkpoint_dir)/model_config.checkpoint_name
        checkpoint=torch.load(load_path)
        classifier.load_state_dict(checkpoint["model"])  

        MPI_testset=MPIDataset(Path(model_config.dataset_dir)/'test_set.csv')    
        MPI_dataloader=DataLoader(
            MPI_testset,
            batch_size=32,
            shuffle=False,
            drop_last=False
        )
        pred,label_true=predict(classifier,MPI_dataloader)
        
        label_sum.append(pred)
        label=label_true
        

    label_sum=[x for x in label_sum if x]
    label_sum=np.array(label_sum)
    label_sum=np.sum(label_sum,axis=0)

    

    ## majority voting:
    majority_label=np.copy(label_sum)
    majority_label[majority_label<(num_classifier/2)]=0
    majority_label[majority_label>(num_classifier/2)]=1
    ### print statistics
    print("Majority Voting:")
    print_stat(majority_label,label)




    ##safetest voting:
    minimum_label=np.copy(label_sum)
    minimum_label[minimum_label>0]=1
    print("Safetest Voting:")
    print_stat(minimum_label,label)


        
        
def print_stat(pred,label):
    average_acc=metrics.accuracy_score(label,pred)
    precision=metrics.precision_score(label, pred) 
    recall=metrics.recall_score(label,np.around(pred).astype(int),average='binary')
    auc=metrics.roc_auc_score(label,pred)
    confusion_mtx=metrics.confusion_matrix(label, pred)
    specificity=confusion_mtx[0,0]/(confusion_mtx[0,0]+confusion_mtx[0,1])
    statistics={'average_acc':average_acc,
    'average_precision':precision,
                'Sensitivity': recall,   ## sensitivity
                'auc': auc,
            'confusion_matrix':confusion_mtx,
            'specificity':specificity
            }
    tabular_report=metrics.classification_report(label,np.around(pred).astype(int))

    print(statistics)
    print(tabular_report)
    disp=metrics.ConfusionMatrixDisplay(statistics['confusion_matrix'])
    disp.plot()





def predict(classifier,data_loader):
    pred=[]
    label=[]
    device= next(classifier.parameters()).device
    for i, data in enumerate(tqdm(data_loader),1):
        mpi_batch=data["image"].to(device)    #put on device            
        with torch.no_grad():
            classifier.eval()
            pred_batch=torch.round(torch.sigmoid(classifier(mpi_batch)))
        pred.extend(pred_batch.data.cpu().numpy())
        label.extend(data["label"].data.cpu().numpy())
    
    pred=[int(i[0]) for i in pred]
    return pred,label
                           
    

if __name__ == "__main__":
    ensemble_model()