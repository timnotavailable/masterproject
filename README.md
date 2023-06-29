# masterproject
All of the clean code in master thesis. The master thesis comprises 4 parts:   First 3 parts are classification , the  last  exciting part is localization.

Origin data is stored in "DUMP" files, which contained .image.bin and .txt files which record patients' information.
All of the .ipynb file is executatble step by step.

We introduce the projects in FFTDL, transfer learning and Localization, configured with  hydra

## 0.Introduction:
Introduction of the file hierachy, take FFTDL as example:
|.FFTDL

 _ DUMP       ---file containing origin data
 
 _ config     ---configuration file (.yaml)
 
 _ xx.py      --- customed py as library or executable
 


## 1. Configurate Environment:
#### pip install -r requirement.txt 

All the packages used have been written in this requirment document, including package version. 


## 2. preprocess the model:
In commmand line 
#### python3 preprocess.py
In a .ipynb file one can type %run preprocess.py

It will generate "metadata" which contains all the directory, a "processed data " containing all the processed data in .pt format.

## 3. train the model:
Example:
#### python3 train_FFC_Res18.py 

train the model, at the same time generate those files:

    training_monitoring :reconding the performance matrix
    
    run/xx_model  :  training curve is recorded for certain model
    
    checkpoints/xx_model: record the checkpoints in case of model interuptting.
    


## 4. test result:
#### python3 test_single.py

test all the model performance.

## 5. model ensemble:
#### python3 model_ensemble.py

ensemble all the models and get the best performance





It is strongly suggested that change the .yaml instead of .py to change the hyperparameter.

In .execution .ipynb file man can see the intermediate training processs and testing result.
