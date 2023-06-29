import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
import random
from tqdm import tqdm

class MPIDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path=Path(csv_path)
        self.metadata=pd.read_csv(csv_path)
        self.file_numbers=len(self.metadata)
     
    def __len__(self):
        return self.file_numbers
    
    def __getitem__(self,index):
        
        ## debug if data is not correct
        MPI_datapath=Path(self.metadata.loc[index]["filepath"])
        MPI_data=torch.load(MPI_datapath)

        return MPI_data

    
def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir, name_prefix='model'):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_name = name_prefix+".ckpt-{}.pt".format(step)
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}\n".format(checkpoint_path.stem))


def load_checkpoint(model, optimizer, scheduler, load_path):
    print(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["step"]

class Evaluator(object):
    def __init__(self):
        """Evaluator.
        Args:
         model: inhereit from object
       """
    def evaluate(self, classifier,data_loader): 
        output_dict={}
        device= next(classifier.parameters()).device
        for i, data in enumerate(tqdm(data_loader),1):
            mpi_batch=data["image"].to(device)    #put on device            
            with torch.no_grad():
                classifier.eval()
                pred_batch=torch.round(torch.sigmoid(classifier(mpi_batch)))
            append_to_dict(output_dict,'pred',pred_batch.data.cpu().numpy())
            append_to_dict(output_dict,'label',data["label"].data.cpu().numpy())
         
        for key in output_dict.keys():
            output_dict[key]=np.concatenate(output_dict[key],axis=0)
            
        pred=output_dict['pred']
        label=output_dict['label']
        ## average=None for multi-classification, otherwise average ="binary " or default
        average_acc=metrics.accuracy_score(label,pred)
        precision=metrics.precision_score(label, pred) 
        recall=metrics.recall_score(label,np.around(pred).astype(int),average='binary')
        auc=metrics.roc_auc_score(label,pred)
        confusion_mtx=metrics.confusion_matrix(label, pred)
        specificity=confusion_mtx[0,0]/(confusion_mtx[0,0]+confusion_mtx[0,1])
#         lrap = metrics.label_ranking_average_precision_score(label, pred)   for multi-classification
        tabular_report=metrics.classification_report(label,np.around(pred).astype(int))
        statistics={'average_acc':average_acc,
            'average_precision':precision,
                      'Sensitivity': recall,   ## sensitivity
                      'auc': auc,
                    'confusion_matrix':confusion_mtx,
                    'specificity':specificity
                   }
        
        return statistics,tabular_report
    
class Multi_Evaluator(object):
    def __init__(self):
        """Evaluator.
        Args:
         model: inhereit from object
       """
    def evaluate(self, classifier,data_loader): 
        output_dict={}
        device= next(classifier.parameters()).device
        for i, data in enumerate(tqdm(data_loader),1):
            mpi_batch=data["image"].to(device)    #put on device            
            age=data["age"].to(device)
            TID=data["TID_INDEX"].to(device)
            with torch.no_grad():
                classifier.eval()
                pred_batch=torch.round(torch.sigmoid(classifier(mpi_batch,age,TID)))
            append_to_dict(output_dict,'pred',pred_batch.data.cpu().numpy())
            append_to_dict(output_dict,'label',data["label"].data.cpu().numpy())
         
        for key in output_dict.keys():
            output_dict[key]=np.concatenate(output_dict[key],axis=0)
            
        pred=output_dict['pred']
        label=output_dict['label']
        ## average=None for multi-classification, otherwise average ="binary " or default
        average_acc=metrics.accuracy_score(label,pred)
        average_precision=metrics.average_precision_score(label, pred) 
        recall=metrics.recall_score(label,np.around(pred).astype(int),average='binary')
        auc=metrics.roc_auc_score(label,pred)
        confusion_mtx=metrics.confusion_matrix(label, pred)
#         lrap = metrics.label_ranking_average_precision_score(label, pred)   for multi-classification
        tabular_report=metrics.classification_report(label,np.around(pred).astype(int))
        statistics={'average_acc':average_acc,
            'average_precision': average_precision,
                      'recall': recall,
                      'auc': auc,
                    'confusion_matrix':confusion_mtx
                   }
        
        return statistics,tabular_report
    
def append_to_dict(dict,key,value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key]=[value]