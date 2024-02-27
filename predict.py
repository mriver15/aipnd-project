# @author mriver15 Michael Rivera
# Uses a trained network to predict the class for an input image
from src import utils
from torchvision import transforms
import torch

import json

def load_model(path):
    """
    Returns model from checkpoint
    @param path path to model checkpoint
    """
    model = torch.load(path)
    
    print("Loaded Model ...")
    # print("Epochs Trained: {}".format(model.epoch))
    # print("Model State Dict: {}".format(model.model_state_dict))
    # print("Optimizer State Dict: {}".format(model.optimizer_state_dict))
    # print("Last loss: {}".format(model.loss))
    # print("Classifier: {}".format(model.classifier))
    return model

def load_image(path, img_transform):
    """
    Loads image and returns a trainable one
    @param path Path to image to load
    @param img_transform Trn 
    """
    with open(path) as img:
        return 

def process_img():
    return -1

def get_transform():
    """
    Builds and returns transform for image training
    """
    return transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

def predict(image, model):
    """
    Predicts image category using model
    @param image image loaded and preprocessed
    @param model model loaded from checkpoint
    """
    return -1

def get_category_mapping(path):
    """
        Retrieves category file
        contains mapping of category ID to the name of the flower category
        @param path path to mapping character
    """
    with open(path, 'r') as f:
        categories = json.load(f)
    return categories

def main():
    args = utils.pred_args()
    print(args)

    pred_transform = get_transform()
    model = load_model(args.checkpoint)


if __name__ == '__main__':
    main()
