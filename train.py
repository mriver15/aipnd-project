# @author mriver15 Michael Rivera
# Trains a network on a dataset and saves the model as a checkpoint.

# Local Resources & Functions
from src import utils

# Import nn + optim plus functional from nn
# makes it easy to access in shorthand, not sure if this will deteriorate performance
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

# Time & Os
import time
from os.path import isdir
from datetime import datetime

# Data Structures
from collections import OrderedDict
import json


def main():
    """
    Ideally orchestrates training and gathers data as required for training to be done
    """
    time_start = time.gmtime()
    print("---- Training Started ----")
    # Retrieves arguments from cli
    args = utils.train_args()
    
    # Training Data
    train_dl, valid_dl = get_dataloaders(args.data)
    dls = {
        "train": train_dl, "valid": valid_dl
    }

    # loads model based on architecture
    if args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("{} Is not a valid Architecture".format(args.arch))
        exit(0)

    # Classifier definition
    model.classifier = get_classifier(args.hidden_units)
    model.criterion = nn.NLLLoss()
    if args.learning_rate:
        model.optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    else:
        model.optimizer = optim.Adam(model.classifier.parameters(), lr=.0025)

def get_classifier(model, hidden_units):
    """
    Retrieves classifier using custom in_units based on model and hidden_units
    @param model Either vgg16 or densenet121
    @param hidden_units preferably a number between 1024 & 180 or a number
        between 25088 & 180
    """
    in_units = 1024
    if model == "vgg16":
        in_units = 25088
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_units, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 180)),
                          ('relu', nn.ReLU()),
                          ('droupout',nn.Dropout(0.5)),
                          ('fc3', nn.Linear(180, 102)),
                          ('relu', nn.ReLU()),
                          ('fc4', nn.Linear(102, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    return classifier



def get_device(device):
    """
    Checks for gpu availability if requested
    @param device cpu or gpu
    """
    if not device:
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_category_mapping():
    """
        Retrieves cat_to_name file
        contains mapping of category ID to the name of the flower category
    """
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def get_dataloaders(img_dir):
    """
    Retrieves and prepares training & validation data images
    @param img_dir image directory for training
    """
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(img_dir, transform=train_transform) 
    valid_dataset = datasets.ImageFolder(img_dir, transform=valid_transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    return train_dataloader, valid_dataloader

def train(model, dataloader, print_frequency, device):
    """
        trains model using dataloader train + validation profiles
        model object carries with it the criterion, optimizer, epochs
        print frequency goes by image processed, device does its part
    """
    
    # Parameter Freezing
    for param in model.parameters():
        param.requires_grad = False

    

    return -1



if __name__ == '__main__':
    main()
