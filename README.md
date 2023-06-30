# masterproject
All of the clean code in master thesis. The master thesis comprises 4 parts:   First 3 parts are classification , the  last  exciting part is localization.

This project is going to be posted to journal " Artificial Intelligence in medicine".

# Highlight (Snipped)

Fourier Unit:

![Fourier Unit](https://github.com/timnotavailable/masterproject/assets/92996426/44ed7fd6-e434-45fc-b2c3-f28367ccedcd)

Local Fourier Unit:

![Local Fourier Unit](https://github.com/timnotavailable/masterproject/assets/92996426/065b876d-9081-49a5-9c9c-78bf8d0652dc)

Localization:

![Localization-algorithm](https://github.com/timnotavailable/masterproject/assets/92996426/913f3afc-0165-4959-9c65-1451689a031a)

![N184382649semi](https://github.com/timnotavailable/masterproject/assets/92996426/984e404a-f98e-48d6-8311-5e85af4e47d3)


# Abstract
In this thesis, it has been firstly introduced a special way to resample the polar map. Then in the next chapter we again regrouped the data and performed Fourier transform. Based on those spatial and Fourier series, we set up a group of statistical estimators and performed hypothesis testing to demonstrate the significant difference between normal and abnormal patients and  superior performance of our statistical estimators over \ac{TID}.

Then, we utilized the previous chapter's conclusion and designed a dedicated neural network based on Resnet. With the help of the 5 models ensemble, we reached a state-of-art agreement(91\%) and specificity (100\%) even with a small dataset. Based on different agency, combined model could acquire different results that we want to trade off.

What's more, we verified the effective clinical parameter and explored which clinical parameters could  be added into neural networks under our patients distribution.

Last but the most pivotal is, we finally explored and set up another localization evaluation system  based on  bayesian statistics in contrast to frequentist one. We compared with conventional \ac{TPD} and analysed the promising advantages but also pitfalls of our new method.

We firmly believed that our work have the potential to become the new generation of evaluation and interpretation of stress and rest polar map with the help of modern deep learning technique.


## 0.Introduction:

Origin data is stored in "DUMP" files, which contained .image.bin and .txt files which record patients' information.
All of the .ipynb file is executatble step by step.

We introduce the projects in FFTDL, transfer learning and Localization, configured with  hydra


Introduction of the file hierachy, take FFTDL as example:

.FFTDL

 |_ DUMP       ---file containing origin data
 
 |_ config     ---configuration file (.yaml)
 
 |_ xx.py      --- customed py as library or executable py 
 


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
