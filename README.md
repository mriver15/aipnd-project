# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Contents

### Part 1 - Development

Considering that the program should be ran in either cpu or gpu, some of the parts in the notebook might be out of order, I consider it to be an exploratoty phase. Towards the end of Part 2, I was able to enable Cuda on my personal laptop gpu and speed up the process of training a model.

A few things to consider going through the notebook:

- Every part on the rubric has been labeled in code segments
- Although some parts aren't being modularized in this implementation, a more concise approach is done on the second part

### Part 2 - Command Line App

#### Setup

Personally ran in RTX GeForce 3060 Laptop

Torch version: ```conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia```

#### Train.py

Basic usage: ```python train.py data_directory```
Prints out training loss, validation loss, and validation accuracy as the network trains
Options: 
* Set directory to save checkpoints: ```python train.py data_dir --save_dir save_directory``` 
* Choose architecture: ```python train.py data_dir --arch "densenet121"``` 
* Set hyperparameters: ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20``` 
* Use GPU for training: ```python train.py data_dir --gpu gpu```

#### Predict.py

Basic usage: ```python predict.py /path/to/image checkpoint```
Options: 
* Return top K most likely classes: ```python predict.py input checkpoint --top_k 3``` 
* Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json``` 
* Use GPU for inference: ```python predict.py input checkpoint --gpu```