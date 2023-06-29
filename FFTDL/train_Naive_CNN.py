import hydra
import hydra.utils
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

##customed dataset and models
from utils import MPIDataset, save_checkpoint, load_checkpoint
from models.Classifiers import Naive_CNN

import random
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1997)

@hydra.main(config_path="config/train",config_name="train_Naive_CNN.yaml")   
def train_model(cfg:DictConfig):
    validation_csv_dir=Path(hydra.utils.to_absolute_path("training_monitoring"))  
    validation_csv_dir.mkdir(exist_ok=True, parents=True)
    validation_csv_path=validation_csv_dir/cfg.model.checkpoint_dir
    
    Classifier=eval(cfg.model.type)
    classifier=Classifier().to(device)
    
    ## model information    
    print("train on ", device)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters())
    print("pytorch total params", pytorch_total_params)
    
    ## tensorboard usage
    writer_logdir=Path(hydra.utils.to_absolute_path("runs"))/"{}".format(cfg.model.type)    
    writer=SummaryWriter(log_dir=writer_logdir,comment=cfg.model.type)
    
    optimizer=optim.AdamW(classifier.parameters(),lr=cfg.optimizer.lr)
    
    ###learning rate deteriotion
    ### gamma:decay parameter, Multiplicative factor of learning rate decay. Default: 0.1.
    ### https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
#     scheduler=torch.optim.lr_scheduler.StepLR(
#         optimizer,
#         step_size=cfg.scheduler.step_size,
#         gamma=cfg.scheduler.gamma
#     )
    ###every checkpoint_interval iteration check performance of model   Read checkpoint:
    checkpoints_patdir=Path(hydra.utils.to_absolute_path("checkpoints"))
    checkpoints_patdir.mkdir(exist_ok=True, parents=True)
     
    checkpoint_dir=checkpoints_patdir/cfg.model.checkpoint_dir
    if cfg.model.load_model:   ## resume training
        load_path=Path(checkpoint_dir)/cfg.model.checkpoint_name
        global_step=load_checkpoint(classifier,optimizer,scheduler,load_path) ## record current checkpoint   load_checkpoint return global_step
    else:
        global_step=0
        # Create evaluation_csv file
        ### save validation metrics
        csv_columns=["global_step"]
        validation_csv=pd.DataFrame(columns=csv_columns)
        validation_csv.to_csv(validation_csv_path,index=False)

    train_dataset=MPIDataset(Path(cfg.training_settings.dataset_dir)/"train_set.csv")  
    val_dataset=MPIDataset(Path(cfg.training_settings.dataset_dir)/"val_set.csv")
    
    train_dataloader=DataLoader(
        train_dataset,
        batch_size=cfg.training_settings.batch_size,
        shuffle=True,
        num_workers=0,
#         num_workers=cfg.training_settings.n_workers,        
#         pin_memory=True,  ## Host to GPU copies are much faster when they originate from pinned 
        drop_last=cfg.training_settings.drop_last)

    val_dataloader=DataLoader(
        train_dataset,
        batch_size=cfg.training_settings.batch_size,
        shuffle=True,
        num_workers=0,
#         num_workers=cfg.training_settings.n_workers,
#         pin_memory=True,
        drop_last=cfg.training_settings.drop_last)

 
    ## define loss function 
    loss_function=nn.BCEWithLogitsLoss()  ## final layer without activated, for numerical stability

    early_stopper=EarlyStopper(patience=cfg.early_stopper.patience,min_delta=cfg.early_stopper.min_delta)  ## could use hydra to configurate patience and min_delta
    evaluator=Evaluator() ## for validation and evaluation   
    n_epochs=cfg.training_settings.n_steps//len(train_dataloader)+1  ## How many steps in 1 epoch? quantity of dataloader numbers in train dataset
    start_epoch=global_step//len(train_dataloader)+1  ## falls interrupted, then could be recovered
    
    for epoch in range(start_epoch,n_epochs+1):
        #-----------------------------------------#
        #train process
        #-----------------------------------------#
        classifier.train()
        train_loss=0
        
        for i,data_batch in enumerate(tqdm(train_dataloader),1):
            mpi_image=data_batch["image"].to(device)
            label_batch=data_batch['label'].to(device)
            label_batch=torch.unsqueeze(label_batch,1)
            
            ## train the classification model:
            logits_batch=classifier(mpi_image)
            
            optimizer.zero_grad()
            loss=loss_function(logits_batch,label_batch.float())
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss+=(loss.item()-train_loss)/i
            
            global_step+=1
            
            ## every  cfg.training_settings.checkpoint_interval steps , save the model 
            if global_step % cfg.training_settings.checkpoint_interval==0:
                statistics,tabular_report=evaluator.evaluate(classifier,val_dataloader)           
                save_checkpoint(classifier,optimizer,scheduler,global_step,checkpoint_dir,name_prefix='model')
                print("training global_step:{}\n".format(global_step))
                print(statistics)
                print(tabular_report)

                #record in evaluation csv
                with open(str(validation_csv_path),'a') as fd:
                    fd.write(str(global_step))
                    fd.write(',' + str(np.mean(statistics['average_acc'])))
                    fd.write(',' + str(np.mean(statistics['average_precision'])))
                    fd.write(',' + str(np.mean(statistics['recall'])))
                    fd.write(',' + str(np.mean(statistics['auc'])))
                    fd.write('\n')
        
                print("training epoch:{}, global loss:{:.3E}\n".format(epoch,train_loss))
            
        #---------------------------------------------------#
        ## val process
        #---------------------------------------------------#
        classifier.eval()
        val_loss=0
        ## used to record correct number
        correct=0
        total_pred=np.zeros(0)
        total_target=np.zeros(0)
            
        with torch.no_grad():
            for data_batch in val_dataloader:
                mpi_image=data_batch["image"].to(device)
                label_batch=data_batch['label'].to(device)
#                 print("label_batch shape is ",label_batch.shape)
#                 print("label_batch dtype is", label_batch.dtype)
                label_batch=torch.unsqueeze(label_batch,1)
#                 print("unsqueezed label_batch shape is ",label_batch.shape)
#                 print("label_batch dtype is", label_batch.dtype)
                logit_batch=classifier(mpi_image) 
#                 print("logit_batch shape is ", logit_batch.shape)
#                 print("logit_batch dtype is ", logit_batch.dtype)
                val_loss+=loss_function(logit_batch,label_batch.float()).item()
                
                pred_batch=torch.round(F.sigmoid(logit_batch))
#                 print("pred_batch dtype is ", pred_batch.dtype)
                total_pred=np.append(total_pred,label_batch.cpu().numpy())
                total_target=np.append(total_target,label_batch.cpu().numpy())
                correct+=pred_batch.eq(label_batch.view_as(pred_batch)).sum().item()
            val_loss/=len(val_dataloader.dataset)
            val_acc=100*correct/len(val_dataloader.dataset)
        #----------------------------------------------#  
        # write in tensor board
        #----------------------------------------------#
        ##https://pytorch.org/docs/stable/tensorboard.html   tensorboard usage

        
        writer.add_scalar('Loss/train loss',train_loss,global_step=global_step)
        writer.add_scalar('Loss/Validation loss',val_loss,global_step=global_step)
        writer.add_scalar('Correct', correct,global_step=global_step)
        writer.add_scalar('val_acc', val_acc,global_step=global_step)
        ## early stopping:
        if early_stopper.early_stop(val_loss):
            statistics,tabular_report=evaluator.evaluate(classifier,val_dataloader)           
            save_checkpoint(classifier,optimizer,scheduler,global_step,checkpoint_dir,name_prefix='model')
            print("training global_step:{}\n".format(global_step))
            print(statistics)
            print(tabular_report)
            
            #record in evaluation csv
            with open(str(validation_csv_path),'a') as fd:
                fd.write(str(global_step))
                fd.write(',' + str(np.mean(statistics['average_acc'])))
                fd.write(',' + str(np.mean(statistics['average_precision'])))
                fd.write(',' + str(np.mean(statistics['recall'])))
                fd.write(',' + str(np.mean(statistics['auc'])))
                fd.write('\n')
            
            print("Early stopping, validation loss cannot decrease anymore, choose current state_dict or previous savepoint to do final test")        
            print("training epoch:{}, global loss:{:.3E}\n".format(epoch,train_loss))
            print("training epoch:{}, val loss:{:.3E}\n".format(epoch,val_loss))
            print("training global_step:{}\n".format(global_step))
            
            
            return

            
                
                   
    



## for classification, binary classify, only images

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
        
## early stopping  
## https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False        
        
        
        
        
## main function
if __name__=="__main__":
    train_model()
      