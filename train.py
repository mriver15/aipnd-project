# @author mriver15 Michael Rivera
# Trains a network on a dataset and saves the model as a checkpoint.
# Test Launch String
# python train.py --data flowers --save_dir runs --arch densenet121 --epochs 1

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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("---- Training Started @ {}----".format(timestamp))
    # Retrieves arguments from cli
    args = utils.train_args()
    
    # Training Data
    dls = get_dataloaders(args.data)
    

    # Preparing to load
    # loads model based on architecture
    if args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("{} Is not a valid Architecture".format(args.arch))
        exit(0)

    # Zero Grad
    # Parameter Freezing for model buildup
    for param in model.parameters():
        param.requires_grad = False

    # Classifier definition
    model.classifier = get_classifier(args.arch,args.hidden_units)

    # Criterion definition
    # Uses NLLLoss as previous attempts
    model.criterion = nn.NLLLoss()


    if args.learning_rate:
        model.optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    else:
        model.optimizer = optim.Adam(model.classifier.parameters(), lr=.0025)

    # move model to GPU/CPU
    # bundle model with device
    model.device = get_device(args.gpu)
    model.to(model.device)

    # Save directory
    model.directory = args.save_dir
    model = train(model, dls, args.print_every, get_epochs(args.epochs))

def get_epochs(epochs):
    """
    Defaults or returns parameter
    Separated for alternate functionality on epochs, perhaps other calculation
    @param epochs amount of runs for training
    """
    if not epochs:
        return 10
    return int(epochs)

def train_epoch(epoch, model, dataloader, print_every):
    """
    trains given model for 1 epoch using dataloader and spits out stats every so often

    @param epoch current epoch running int
    @param model training model, comes bundled with optimizer, criterion and more
    @param dataloader train + validation data
    @param print_every int determining every how many images to print out accuracy + loss
    """
    running_loss = 0.0
    last_loss = 0.0

    running_acc = 0.0
    epoch_acc = 0.0

    if not print_every:
        print_every = 10

    model.train()


    for i,data in enumerate(dataloader['train']['dataloader']):
        # Prep Inputs, Labels from data
        # Move to device targetted in model
        inputs,labels = data
        inputs,labels = inputs.to(model.device), labels.to(model.device)

        # Zero gradients
        model.optimizer.zero_grad()

        # Predictions for Batch
        outputs = model.forward(inputs)

        # Compute Loss calculate backpropagation
        loss = model.criterion(outputs, labels)
        loss.backward()

        # Accuracy Calculation
        ps = torch.exp(outputs).data
        print('{} psmax'.format(labels.data))
        equals = (labels.data == ps.max(1)[0]) # Top Result
        running_acc += equals.type_as(torch.FloatTensor()).mean()

        # Adjust learning weights
        model.optimizer.step()

        # Loss
        running_loss += loss.item()

        if i % print_every == 0:
            last_loss = running_loss / print_every
            running_loss = 0
            print("Epoch: {} --- Batch: {} --- Accuracy: {} --- Loss: {}".format(epoch, i,running_acc, last_loss))
    else:
        epoch_acc = running_acc / len(dataloader['train']['dataloader'])
    return model, last_loss, epoch_acc



def get_classifier(model_type, hidden_units):
    """
    Retrieves classifier using custom in_units based on model and hidden_units
    @param model Either vgg16 or densenet121
    @param hidden_units preferably a number between 1024 & 180 or a number
        between 25088 & 180
    """
    in_units = 1024
    if model_type == "vgg16":
        in_units = 25088

    if not hidden_units:
        hidden_units = 512
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
    # classifier = nn.Sequential(OrderedDict([
    #                       ('fc1', nn.Linear(1024, 512)),
    #                       ('relu', nn.ReLU()),
    #                       ('fc2', nn.Linear(512, 180)),
    #                       ('relu', nn.ReLU()),
    #                       ('droupout',nn.Dropout(0.5)),
    #                       ('fc3', nn.Linear(180, 102)),
    #                       ('relu', nn.ReLU()),
    #                       ('fc4', nn.Linear(102, 102)),
    #                       ('output', nn.LogSoftmax(dim=1))
    #                       ]))
    
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
    dataloader = {'train': {}, 'valid': {}}
    dataloader['train']['transform'] = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    dataloader['valid']['transform'] = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    dataloader['train']['dataset'] = datasets.ImageFolder(img_dir, transform=dataloader['train']['transform']) 
    dataloader['valid']['dataset'] = datasets.ImageFolder(img_dir, transform=dataloader['valid']['transform'])

    dataloader['train']['dataloader'] = torch.utils.data.DataLoader(dataloader['train']['dataset'], batch_size=64,shuffle=True)
    dataloader['valid']['dataloader'] = torch.utils.data.DataLoader(dataloader['valid']['dataset'], batch_size=32)

    return dataloader


def train(model, dataloader, print_every, epochs):
    """
        trains model using dataloader train + validation profiles
        model object carries with it the criterion, optimizer, epochs
        print frequency goes by image processed, device does its part
        @param model prepared model, contains multiple attached parts in namespace
        @param dataloader train, validation dataloaders
        @param print_every frequency of status printouts
        @param epochs amount of epochs to train model
    """
    # Tracking variables
    best_vloss = 1_000_000
    running_vloss = 0.0
    
    val_acc = 0
    running_vacc = 0


    # Master train loop
    for i in range(epochs):
        print("Starting Epoch Training --- {}".format(i))

        # Training Step
        model.train()
        model, avg_loss, epoch_acc = train_epoch(i, model, dataloader, print_every)

        running_vloss = 0
        # Validation Step
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for i, vdata in enumerate(dataloader['valid']['dataloader']):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(model.device), vlabels.to(model.device)
                voutputs = model.forward(vinputs)
                vloss = model.criterion(voutputs, vlabels)
                running_vloss += vloss

                # Validation Accuracy
                ps = torch.exp(voutputs).data
                vequals = (vlabels.data == ps.max(1)[0])
                running_vacc += vequals.type_as(torch.FloatTensor()).mean()
            else:
                val_acc = running_vacc / len(dataloader['valid']['dataloader'])
                avg_vloss = running_vloss / (i+1)
        
        
        print("Epoch: {} .. Train Loss: {} .. Valid Loss: {} .. Train Accuracy: {} .. Valid Accuracy: {}".format(i+1, avg_loss, avg_vloss, epoch_acc, val_acc))
        # Upon encountering a better model, save checkpoint
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model.class_to_idx = dataloader['train']['dataset'].class_to_idx
            model.model_path = model.directory + '/' + 'model_{}_{}'.format(datetime.now().strftime('%Y%m%d_%H%M'), i+1)
            torch.save({
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'loss': best_vloss
            }, model.model_path)

    return model


if __name__ == '__main__':
    main()
