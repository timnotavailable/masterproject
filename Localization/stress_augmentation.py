import os
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path 
from tqdm import tqdm
# import system

import warnings
warnings.filterwarnings("ignore")


def augmentation(aug=1,metadata_dir="metadata//stress",processed_dir="processed data//stress",augall=False):
    metadata_dir=Path(metadata_dir)
    processed_dir=Path(processed_dir)
    traincsv_dir=metadata_dir/"train_set.csv"
    train_meta=pd.read_csv(traincsv_dir)
    aug_dir=metadata_dir/"aug_set.csv"
    aug_csv=pd.DataFrame(columns=["filepath"])
    aug_csv.to_csv(aug_dir,index=False)
    
    
    for i in range(len(train_meta["filepath"])):
        num_aug=aug  
        
        patient_pat=train_meta.loc[i]["filepath"].split('\\')
        patient=patient_pat[-1]
        data=torch.load("processed data\\"+"stress\\"+patient)
        if augall==False:
            if data["label"]==1:            
                while num_aug>0:
                    arr_aug=augment_gauss(data["image"])
                    aug_data_pair={
                        "image":arr_aug,                    
                        "label":data['label'],  
                        "scar":data['scar'],
                        "ischemia":data['ischemia'] ,
                        "normal":data['normal'],
                        "SRD_STRESS_DIST":data['SRD_STRESS_DIST']
                    }
                    aug_pair_name="aug"+str(num_aug)+patient             
                    torch.save(aug_data_pair,processed_dir/aug_pair_name)

                    with open(str(aug_dir),'a') as fd:
                        augname='..\\..\\..\\'+ 'processed data\\'+"stress\\"+aug_pair_name
                        fd.write(augname+'\n')                          
                    num_aug-=1
        if augall==True:
                while num_aug>0:
                    arr_aug=augment_gauss(data["image"])
                    aug_data_pair={
                        "image":arr_aug,                    
                        "label":data['label'],  
                        "scar":data['scar'],
                        "ischemia":data['ischemia'] ,
                        "normal":data['normal'],
                        "SRD_STRESS_DIST":data['SRD_STRESS_DIST']
                    }
                    aug_pair_name="aug"+str(num_aug)+patient             
                    torch.save(aug_data_pair,processed_dir/aug_pair_name)

                    with open(str(aug_dir),'a') as fd:
                        augname='..\\..\\..\\'+ 'processed data\\'+aug_pair_name
                        fd.write(augname+'\n')                          
                    num_aug-=1            
    aug_meta=pd.read_csv(aug_dir)
    ## merge dataset
    train_meta=pd.concat([train_meta,aug_meta],axis=0)
    ## shuffle trainset
    train_meta=train_meta.sample(frac=1)                                     ## save and overwrite already saved csv
    train_meta.to_csv(traincsv_dir,mode='w+')
                                                        

     
    
    
def augment_gauss(arr,mean=0,sigma=0.005):
    noise=np.random.normal(mean,sigma,arr.shape)
    arr+=noise
    return arr                                                                                                          
if __name__ == "__main__":
#     augmentation(sys.argv[1], sys.argv[2],sys.argv[3])
    augmentation()   
                                                        
                                                