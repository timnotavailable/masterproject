from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path
import torch
import torch.nn as nn

from models.Classifiers import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import skimage.transform
import copy
import os

from ReadImageBin  import ImageBin,rescale_transform
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                      


def percentile_report(stress_cam224,rest_cam224,pat_name:str):
    """
    Args:
      stress_cam: (1,244,244)
      rest_cam: (1,244,244)
      pat_name: string
   return: tuple->
      rest_percent,ischemia_percent,normal_percent
     
    """
    ## to 15*36 and normalize
    stress_cam=back_transform(stress_cam224[0,:])
    rest_cam=back_transform(rest_cam224[0,:])
  
    
    ## get TID value and determine threshold
    pat_bin,pat_txt=sort_txt_bin("DUMP")
    this_txt=findtxt(pat_name,pat_txt)
    txt_df=pd.read_csv(("DUMP//"+this_txt),sep=" :=",index_col=0,header=None)                     
    TID=np.float32(txt_df.loc["TID_INDEX"])
    ctr_stress=np.float32(txt_df.loc["CTR_VOLUMES(00)"])
    ## create binary  images     
    stress_per=70    ## stress percentile
    rest_per=75     ## rest percentile
    scar_per=90    ## scar percentile    ## configurable
    
    if TID>=110 and ctr_stress>152:
        stress_per-=25
        rest_per-=15
        scar_per-=5
   
    if stress_cam.sum()>0:  ## in case str_thre==0 , numerical stability    
        str_thre=np.percentile(stress_cam[stress_cam>0],stress_per)
    else: str_thre=1
        
    if rest_cam.sum()>0:  ## in case rest_thre==0
        rest_thre=np.percentile(rest_cam[rest_cam>0],rest_per)
        scar_thre=np.percentile(rest_cam[rest_cam>0],scar_per)
    else:
        rest_thre=1 
        scar_thre=1
    
    bi_stress=copy.deepcopy(stress_cam)
    bi_rest=copy.deepcopy(rest_cam)
    bi_scar=copy.deepcopy(rest_cam)
 

    
    bi_stress[bi_stress>=str_thre]=1
    bi_stress[bi_stress<str_thre]=0
    bi_rest[bi_rest>=rest_thre]=1
    bi_rest[bi_rest<rest_thre]=0
    overlap=bi_stress*bi_rest
    bi_scar[bi_scar>=scar_thre]=1
    bi_scar[bi_scar<scar_thre]=0    
    
    bi_ischemia=bi_stress-overlap
    
    ##RGB segmentation
    label_img_rgb=label_img_to_rgb(bi_stress,bi_scar)
#     bi_rest/=2  ## normalize to one because in the previous it*2
   
    
    ##get area map
    path_bin="DUMP//"+pat_name+".image.bin"
    arr4=ImageBin(path_bin).combine_4()
    area_stress=arr4[2,:,:]
    area_rest=arr4[3,:,:]

    
    ## Report of percentile:
    stress_percent=(area_stress*bi_stress).sum()/area_stress.sum()
    rest_percent=(area_rest*bi_rest).sum()/area_rest.sum()    
    ischemia_percent=((bi_ischemia*((area_rest+area_stress)/2)).sum()/((area_rest+area_stress)/2).sum())/2
    scar_percent=(area_rest*bi_scar).sum()/area_rest.sum() 
    
    if str_thre==0:    ## in case all the region seems normal
        stress_percent=0
    if rest_thre==0:    ## in case all the region seems normal
        rest_percent=0

    
    print("abnormal region of stress map is {:6.4f}% " .format(stress_percent*100))
    print("abnormal region of rest map  is {:6.4f}%  " .format(rest_percent*100))
    
    print("region of ischemia is {:6.4f}%".format(ischemia_percent*100))
    print("region of scar is {:6.4f}%".format(scar_percent*100))
    
#     print("region of normal is {:6.4f}%").format(normal_percent*100)
    
    
    ## Visualization:
    fig,axs=plt.subplots(nrows=1,ncols=3,figsize=(12,8))
    
    axs[0].imshow(stress_cam,cmap="Reds")
    axs[0].set_title("Stress Activation Heatmap")
    axs[0].axis('off')
    
    axs[1].imshow(rest_cam,cmap="Reds")
    axs[1].set_title("Rest Activation Heatmap")
    axs[1].axis('off')
    
    axs[2].imshow(label_img_rgb)
    axs[2].set_title("Normal: white; Ichemia : blue ;Scar: red ",wrap=True)
    axs[2].set_xticks(ticks=np.arange(0,36,2))
    axs[2].set_yticks(ticks=np.arange(0,15,1))
    
    fig.tight_layout()
    
    return stress_percent,rest_percent,ischemia_percent,scar_percent
      


def back_transform(img):
    """gray scale class activation map from 224 *224 to 15*36 to match the origin images"""
    bin_image=skimage.transform.resize(img,(15,36))
    bin_image/bin_image.max()
    return bin_image

SEG_LABELS_LIST=[
        {"id":0,"name":"normal",    "rgb_values":[255,255,255]}, ## white
        {"id":1,"name":"stress",    "rgb_values":[0,0,255]},     ## blue  --ischemia
        {"id":2,"name":"rest",    "rgb_values":[255,0,0]},       ## red --scar
        {"id":3,"name":"overlap",    "rgb_values":[0,255,0]}     ## green  ---fixed tissue      
    ]

def label_img_to_rgb(bi_stress,bi_rest):
    bi_overlap=bi_stress*bi_rest
    ## complementary of stress region
    bi_stress=bi_stress-bi_overlap

    bi_rest*=2
    label_img=bi_stress+bi_rest
    labels=np.unique(label_img)
    label_infos=[l for l in SEG_LABELS_LIST if l['id'] in labels]
    label_img_rgb=np.array([label_img,
                           label_img,
                           label_img]).transpose(1,2,0)
    
    for l in label_infos:
        mask=label_img==l['id']
        label_img_rgb[mask]=l['rgb_values']
    
    return label_img_rgb.astype(np.uint8)



def gray_scale(model,input_tensor,target_layers,eigen_smooth=False, aug_smooth=False,method=0,reshape_transform=False):
    """
    Args:
     model:must load the state_dict
     input_tensor: 
     target_layers:usually last layer ,e.g.  model.model.layer4
     
     eigen_smooth:default False
     aug_smooth:default False
     method: default 0 -> GradCAM,  1->GradCAMPlusPlus,2 ->AblationCAM,3 ->HiResCAM
   return:
     gray_scale images: (1,224,224)
     """
    methods={0:"GradCAM",
          1:"GradCAMPlusPlus",
          2:"AblationCAM",
          3:"HiResCAM"
            }
    CAM=eval(methods[method])
    if reshape_transform :
        cam=CAM(model=model,target_layers=target_layers,use_cuda=True,reshape_transform=fc_transform)  
    else :
        cam=CAM(model=model,target_layers=target_layers,use_cuda=True) 
    targets=[BinaryClassifierOutputTarget(1)] 
      
    grayscale_cam=cam(input_tensor=input_tensor,targets=targets)  
    return grayscale_cam


def fc_transform(x):
    return x[0]


def input_tensor():
    """
    input:
      patient_pt: string array which contain pt.files, whose stress and rest are all abnormal
    return:
      stress_input, rest_input: arrays which contain stress and rest input tensors
    """
    patient_pt=pair_abnormal()
    stress_input=[]
    rest_input=[]
    for i in patient_pt:
        stress_name=r"processed data/stress/stress "+i
        rest_name=r"processed data/rest/rest "+i
        stress_input_tensor=torch.from_numpy(torch.load(stress_name)["image"]).unsqueeze(0).to(device)        
        rest_input_tensor=torch.from_numpy(torch.load(rest_name)["image"]).unsqueeze(0).to(device)
        stress_input.append(stress_input_tensor)
        rest_input.append(rest_input_tensor)    
    
    return stress_input,rest_input,patient_pt 
    
def pair_abnormal():
    patient_pt=[]
    for i in os.listdir(r"processed data/rest"):
        name=r"processed data/rest/"+i
        if torch.load(name)["label"]==1:
            rest_pt=i.split()[-1]
            patient_pt.append(rest_pt)
    patient_pt=list(dict.fromkeys(patient_pt))   ###  removing duplicated term , such as augmented terms
    return patient_pt


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