"""

"""
import skimage.transform
import os
import random
import numpy as np
import pandas as pd
import torch
import pathlib
from pathlib import Path 
import contextlib
import joblib
import joblib.parallel
from joblib import delayed, Parallel
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import warnings
warnings.filterwarnings("ignore")

from ReadImageBin  import ImageBin,rescale_transform


## lack of encoding patient sex and code  it in preprocess function!!!





@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

        
@hydra.main(config_path="config",config_name="preprocess_default.yaml")
def preprocess_parallel(cfg:DictConfig):
    ## create csv file for extracted metadata:
    stress_metadata = pd.DataFrame(columns=["filepath"])
    rest_metadata = pd.DataFrame(columns=["filepath"])
    
    metadata_dir=Path(cfg.metadata_dir)
    stress_metadata_dir=metadata_dir/"stress"
    rest_metadata_dir=metadata_dir/"rest"

    stress_metadata_dir.mkdir(exist_ok=True, parents=True)
    rest_metadata_dir.mkdir(exist_ok=True, parents=True) 
    
    stress_meta_csvpath=stress_metadata_dir/"stress_metadata.csv"
    rest_meta_csvpath=rest_metadata_dir/"rest_metadata.csv"
    
    stress_metadata.to_csv(stress_meta_csvpath,index=False)
    rest_metadata.to_csv(rest_meta_csvpath,index=False)
   
    ## number of paired elements in each file
    pat_bin,pat_txt,pat_name=pair_bin_txt(Path(cfg.raw_datadir))
    #### Run parallel jobs, tqdm tracking
    with tqdm_joblib(tqdm(desc="Reading and Labelling", total=len(pat_name))):
        Parallel(n_jobs=-1)(delayed(preprocess)
                    (cfg.raw_datadir, cfg.processed_data, cfg.metadata_dir,cfg.aug) for i in range(len(pat_name)))
    print("Preprocessing data finished! \n")
    
    ##drop out duplicated terms
    stress_metadata_csv=pd.read_csv(stress_meta_csvpath)
    stress_metadata_csv=stress_metadata_csv.drop_duplicates()
    rest_metadata_csv=pd.read_csv(rest_meta_csvpath)
    rest_metadata_csv=rest_metadata_csv.drop_duplicates()    
    ## split train,valid,and test set  [0.8,0.1,0.1 ]or [0.6,0.2,0.2]  for stress data
    train,val,test=np.split(stress_metadata_csv.sample(frac=1,random_state=1997),[int(.6*len(stress_metadata_csv)),int(.8*len(stress_metadata_csv))])
    train_path=stress_metadata_dir/"train_set.csv"
    train.to_csv(train_path,index=False)
    val_path=stress_metadata_dir/"val_set.csv"
    val.to_csv(val_path,index=False)
    test_path=stress_metadata_dir/"test_set.csv"
    test.to_csv(test_path,index=False)
    
    ## get rid of duplicated term
    stress_metadata_csv.to_csv(stress_meta_csvpath,index=False)
    ## split for rest data     
    train,val,test=np.split(rest_metadata_csv.sample(frac=1,random_state=1997),[int(.6*len(stress_metadata_csv)),int(.8*len(stress_metadata_csv))])
    train_path=rest_metadata_dir/"train_set.csv"
    train.to_csv(train_path,index=False)
    val_path=rest_metadata_dir/"val_set.csv"
    val.to_csv(val_path,index=False)
    test_path=rest_metadata_dir/"test_set.csv"
    test.to_csv(test_path,index=False)
    
    ## get rid of duplicated term
    rest_metadata_csv.to_csv(rest_meta_csvpath,index=False)        
        
        

def preprocess(raw_datadir,processed_dir,metadata_dir,aug=0):
    """
    
    """
    ## Step 0: setup processed data directory
    processed_dir=Path(processed_dir)
    processed_dir.mkdir(exist_ok=True,parents=True)
    stress_dir=processed_dir/"stress"
    rest_dir=processed_dir/"rest"
    stress_dir.mkdir(exist_ok=True,parents=True)
    rest_dir.mkdir(exist_ok=True,parents=True)
## Use absolute path to cope with, don't use relative path,shit relative path

    raw_datadir=Path(raw_datadir).resolve()
    
    ## find the paired bin, txt and the patients
    pat_bin,pat_txt,pat_name=pair_bin_txt(raw_datadir)

    
    stress_metadata_csv_dir=Path(metadata_dir)/"stress"/"stress_metadata.csv"
    rest_metadata_csv_dir=Path(metadata_dir)/"rest"/"rest_metadata.csv"

    
    for i in range(len(pat_bin)):
        #step 0 : aug is the number that every sample augment ,default is 0,num_aug is a counter, every time it augmented, it will reduce 1
        num_aug=aug
        ## step 1: process the image bin, here extract 4 channels ,they are pm1,pm2,apm1,apm2
        arr=ImageBin(raw_datadir/"{}".format(pat_bin[i])).combine_4()
        
    
    ## Step 2: 
    ### Step 2.1: find the corresponding txt which matched the image bin file(ensure that they are the data from same patients)
        txtname=findtxt(pat_name[i],pat_txt)
        txt_path=raw_datadir/"{}".format(pat_txt[i])
    ### Step 2.2: process the txt to extract label and other clinical information
        txt_df=pd.read_csv(raw_datadir/txtname,sep=" :=",index_col=0,header=None)
        
        
    ## define  threshold for normal/abnormal ,if stress or rest dict <=90 then it's abnormal
        SRD_STRESS_DIST=float(txt_df.loc["SRD_STRESS_DIST(03)"])
        SRD_REST_DIST=float(txt_df.loc["SRD_REST_DIST(03)"])
        scar=float(txt_df.loc["SRD_ISCHEMIA_DIST(02)"])
        ischemia=np.float32(txt_df.loc["SRD_ISCHEMIA_DIST(01)"])
        normal=np.float32(txt_df.loc["SRD_ISCHEMIA_DIST(00)"])
        TID=np.float32(txt_df.loc["TID_INDEX"])
        
        h,w=arr[0,:,:].shape
        array_stress=np.float32(arr[0,:,:])        
        array_stress=skimage.transform.resize(array_stress,(224,224)) #bilinear intepolation
        array_rest=np.float32(arr[1,:,:])        
        array_rest=skimage.transform.resize(array_rest,(224,224)) #bilinear intepolation
        ## TID >120  hypoperfused or triple vessle disease.
        if SRD_STRESS_DIST<76  :
            stress_label=1    # Abnormal
        else:
            stress_label=0    #normal
        if SRD_REST_DIST<82 or TID>125 :
            rest_label=1    # Abnormal
        else:
            rest_label=0    #normal       
     ##incorporate the array and other clinical information in an dict
        stress_data_pair={
            "image":np.stack([array_stress,array_stress,array_stress],axis=0),
            "label":stress_label,  
            "scar":scar,
            "ischemia":ischemia ,
            "normal":normal,
            "SRD_STRESS_DIST":SRD_STRESS_DIST
        }
        rest_data_pair={
            "image":np.stack([array_rest,array_rest,array_rest],axis=0),
            "label":rest_label,  
            "scar":scar,
            "ischemia":ischemia ,
            "normal":normal,
            "SRD_REST_DIST":SRD_REST_DIST
        }
        
        ## save the data pairs into torch tensor
        stress_pair_name="stress "+pat_name[i]+".pt"
        torch.save(stress_data_pair,stress_dir/stress_pair_name)
        rest_pair_name="rest "+pat_name[i]+".pt"
        torch.save(rest_data_pair,rest_dir/rest_pair_name)
        ## save the directory into metadata.csv file
# "r" - Read - Default value. Opens a file for reading, error if the file does not exist
# "a" - Append - Opens a file for appending, creates the file if it does not exist
# "w" - Write - Opens a file for writing, creates the file if it does not exist
# "x" - Create - Creates the specified file, returns an error if the file exist
# In addition you can specify if the file should be handled as binary or text mode
# "t" - Text - Default value. Text mode
# "b" - Binary - Binary mode (e.g. images)


# lock.acquire()   

        with open(str(stress_metadata_csv_dir),'a') as fd:
            fd.write(str(stress_dir/stress_pair_name)+'\n')
        with open(str(rest_metadata_csv_dir),'a') as fd:
            fd.write(str(rest_dir/rest_pair_name)+'\n')
# lock.release()
        
        
        
def sort_txt_bin(path):
    """
    input:path is a string, which is the directory of the files (image bins and txt)
    :return:image_bin: a list containing name of all image bins  ( not directory included)
         txt: a list containing name of all txt names
    """
    all_file=os.listdir(path)  ## return the names of all files, collected in a list
    image_bin=[]
    txt=[]
    for filename in all_file:
        if filename.endswith('.txt'):
            txt.append(filename)
        elif filename.endswith('.bin') :
            image_bin.append(filename)
        else: continue   
     
    
    return image_bin,txt



def findtxt(target,scope):
    """ this is to do the string match, match:
        target: the patient number you want to find, should be a string , such as N152934550
        scope: the txt list you want to find in this scope 
        
        return:name of matched txt, should be a string
         if not found, return ValueError
    """
    count=0
    for txt in scope:
        pos=txt.find(target)
        if pos==0:
            return scope[count]
        else: count+=1

    raise ValueError('You did not setup the right scope or target')

def pair_bin_txt(path):
    """
    Purpose:
       this is to make pair of image.bin file and .txt, because not all the files are paired, so this function act as filter to select the data could be utilized.
    Args:
       image_bin: a list which contains all the image_bin names
       txt: a list which contains all the txt file names.
   return :
       paired image_bin, paired txt. patients name
    
    """
    image_bin,txt=sort_txt_bin(path)
    ## patients_bin ans_d patients_txt are lists for patients number, containing  N152934550 etc.
    patients_bin=[]
    patients_txt=[]
    ##form: ['N152934550.image.bin'.....##
    for patient in image_bin:
        patient_trun=patient.split('.')
        patients_bin.append(patient_trun[0])
    ##form : ['N152934550_2020.11.16_anon.txt'.....]##
    for patient in txt:
        patient_trun=patient.split('_')
        patients_txt.append(patient_trun[0])
        
    ##convert to set and find intersection of two sets
    bin_set=set(patients_bin)
    txt_set=set(patients_txt)
    bin_txt_set=bin_set.intersection(txt_set)
    
    ## bin set and txt file
    final_bin=[]
    final_txt=[]
    patients_name=list(bin_txt_set)  ## intersection
    
    for patient in patients_name:
        #first concatenate  image.bin
        patient_bin=patient+'.'+'image'+'.'+'bin'
        final_bin.append(patient_bin)
        
        ## concatenate  txt:
        patient_txt=findtxt(patient,txt)
        final_txt.append(patient_txt)
    
    return final_bin, final_txt,patients_name
    

if __name__ == "__main__":
    preprocess_parallel()