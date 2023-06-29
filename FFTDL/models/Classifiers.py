import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .model_zoo.ffc_resnet import *



class Naive_CNN(nn.Module):
    """ CNN net"""
    def __init__(self):
        super().__init__()
        self.cnn_model=nn.Sequential(
            nn.Conv2d(4,10,kernel_size=3,stride=1,padding=1),   ## 36*15
            nn.MaxPool2d(kernel_size=(3,2),stride=(3,2)),     ## 12*7
            nn.ReLU(),
            nn.Conv2d(10,20,kernel_size=(3,2),stride=1,padding=(1,0)),  ##12*6
            nn.MaxPool2d(kernel_size=2,stride=2),  ## 6*3
            nn.ReLU()
        )
        self.fcnn=nn.Sequential(
            nn.Linear(20*2*8,15),
            nn.ReLU(),
            nn.Linear(15,1)          
        )
    def forward(self,x):

        ## cnn model
        x=self.cnn_model(x)
#         print(x.shape)  #torch.Size([32, 20, 2, 8])

        ## flatten
        x = x.view(x.shape[0], -1)
        ## fcnn
        x=self.fcnn(x)
        
        return x

    
class TinyCNN(nn.Module):
    def __init__(self, image_channels: int, num_classes: int, dropout: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=3, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.3) if dropout else None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    
##ResNet Zoo

class Res18(nn.Module):
    "Res 18 net, without clinical information"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=models.resnet18(num_classes=1)
        self.model.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self,x):
        return self.model(x)
        
class Res34(nn.Module):
    "Res 34 net, without clinical information"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=models.resnet34(num_classes=1)
        self.model.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self,x):
        return self.model(x)

class Res50(nn.Module):
    "Res 50 net, without clinical information"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=models.resnet50(num_classes=1)
        self.model.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self,x):
        return self.model(x)
    
class Res101(nn.Module):
    "Res 101 net, without clinical information"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=models.resnet101(num_classes=1)
        self.model.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self,x):
        return self.model(x)
class Res152(nn.Module):
    "Res 152 net, without clinical information"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=models.resnet152(num_classes=1)
        self.model.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self,x):
        return self.model(x)
    
# FFT zoo
#https://github.com/pkumivision/FFC

class FFC_res18(nn.Module):
    "FFC_res 18 net, without clinical information"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=ffc_resnet18(num_classes=1)
        self.model.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.maxpool=nn.Identity()
        
    def forward(self,x):
        return self.model(x)

class FFC_res34(nn.Module):
    "FFC_res 34 net, without clinical information"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=ffc_resnet34(num_classes=1)
        self.model.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.maxpool=nn.Identity()
        
    def forward(self,x):
        return self.model(x)    

    
class FFC_res50(nn.Module):
    "FFC_res 50 net, without clinical information"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=ffc_resnet50(num_classes=1)
        self.model.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.maxpool=nn.Identity()
        
    def forward(self,x):
        return self.model(x) 
    
class FFC_res26(nn.Module):
    "FFC_res 26 net, without clinical information"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=ffc_resnet26(num_classes=1)
        self.model.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.maxpool=nn.Identity()
        
    def forward(self,x):
        return self.model(x)
    
class FFC_res101(nn.Module):
    "FFC_res 101 net, without clinical information"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=ffc_resnet101(num_classes=1)
        self.model.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.maxpool=nn.Identity()
        
    def forward(self,x):
        return self.model(x)    
    

    ###     state of art  (super big model):

##EfficientNet V2 Zoo:
#http://pytorch.org/vision/main/models/efficientnetv2.html
#https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

class Efficientnet_v2_s(nn.Module):
    "Efficientnet_v2_s, with only images"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=models.efficientnet_v2_s()
        self.model.features[0]=nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1]=nn.Linear(in_features=1280, out_features=1, bias=True)        
        
    def forward(self,x):
        return self.model(x)

class Efficientnet_v2_m(nn.Module):
    "Efficientnet_v2_s, with only images"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=models.efficientnet_v2_m()
        self.model.features[0]=nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1]=nn.Linear(in_features=1280, out_features=1, bias=True)
    def forward(self,x):
        return self.model(x)

class Efficientnet_v2_l(nn.Module):
    "Efficientnet_v2_s, with only images"
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams=hparams
        self.model=models.efficientnet_v2_l()
        self.model.features[0]=nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1]=nn.Linear(in_features=1280, out_features=1, bias=True) 
    def forward(self,x):
        return self.model(x)       
        
## ViT



###Multi-modal detection

## transfer learning: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html?highlight=transfer%20learning
class mmd_Naive_CNN(Naive_CNN):
    def __init__(self,pretrained_path=None,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.parameters():
            param.requires_grad=False ## .requires_grad_()  to freeze and do transfer learning
        self.fcnn[2]=nn.Linear(15,3*alpha) ## change last layer to imaging features 
        for param in self.fcnn[2].parameters():
            param.requires_grad=True  ## last layer remain to be learnt
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.cnn_model(x)
        x=self.fcnn(x)
        img_vec=x.view(x.shape[0],-1)
        
        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)

        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred

class mmd_Res18(Res18):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.fc=nn.Linear(512,3*alpha)
        for param in self.model.fc.parameters():
            param.requires_grad=True
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)
        #-----
        #-------
        #--------- above should be change, other direct copy
        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred

class mmd_Res34(Res34):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.fc=nn.Linear(512,3*alpha)
        for param in self.model.fc.parameters():
            param.requires_grad=True
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred
        
class mmd_Res50(Res50):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.fc=nn.Linear(512,3*alpha)
        for param in self.model.fc.parameters():
            param.requires_grad=True
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred   
class mmd_Res101(Res101):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.fc=nn.Linear(512,3*alpha)
        for param in self.model.fc.parameters():
            param.requires_grad=True
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred     
class mmd_Res152(Res152):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.fc=nn.Linear(512,3*alpha)
        for param in self.model.fc.parameters():
            param.requires_grad=True
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred 

class mmd_FFC_res18(FFC_res18):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.fc=nn.Linear(512,3*alpha)
        for param in self.model.fc.parameters():
            param.requires_grad=True
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred 

class mmd_FFC_res34(FFC_res34):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.fc=nn.Linear(512,3*alpha)
        for param in self.model.fc.parameters():
            param.requires_grad=True
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred 
class mmd_FFC_res50(FFC_res50):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.fc=nn.Linear(512,3*alpha)
        for param in self.model.fc.parameters():
            param.requires_grad=True
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred 
class mmd_FFC_res26(FFC_res26):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.fc=nn.Linear(512,3*alpha)
        for param in self.model.fc.parameters():
            param.requires_grad=True
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred 
class mmd_FFC_res101(FFC_res101):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.fc=nn.Linear(512,3*alpha)
        for param in self.model.fc.parameters():
            param.requires_grad=True
        self.head=nn.Linear(3*alpha+3,1)
        
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred 
    
class mmd_Efficientnet_v2_s(Efficientnet_v2_s):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.parameters():
            param.requires_grad=False
        self.model.classifier[1]=nn.Linear(in_features=1280, out_features=3*alpha, bias=True)
        for param in self.model.classifier[1].parameters():
            param.requires_grad=True 
        self.head=nn.Linear(3*alpha+3,1)
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred
class mmd_Efficientnet_v2_m(Efficientnet_v2_m):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.parameters():
            param.requires_grad=False
        self.model.classifier[1]=nn.Linear(in_features=1280, out_features=3*alpha, bias=True)
        for param in self.model.classifier[1].parameters():
            param.requires_grad=True 
        self.head=nn.Linear(3*alpha+3,1)
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred 
class mmd_Efficientnet_v2_l(Efficientnet_v2_l):
    def __init__(self,pretrained_path,alpha=1):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.alpha=alpha
        for param in self.parameters():
            param.requires_grad=False
        self.model.classifier[1]=nn.Linear(in_features=1280, out_features=3*alpha, bias=True)
        for param in self.model.classifier[1].parameters():
            param.requires_grad=True 
        self.head=nn.Linear(3*alpha+3,1)
    def forward(self,x,age,TID,sr_ratio):
        x=self.model(x)
        img_vec=x.view(x.shape[0],-1)

        age=torch.unsqueeze(age,1)
        TID=torch.unsqueeze(TID,1)
        sr_ratio=torch.unsqueeze(sr_ratio,1)    
        mul_vec=torch.cat((img_vec,age,TID,sr_ratio),1)        
        mul_vec=F.relu(mul_vec)
        pred=self.head(mul_vec)
        return pred       