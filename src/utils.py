'''
@author mr675v  Michael Rivera
Holds meta functions, used by predict.py & train.py
'''
import argparse


def train_args():
    '''
    Retrieves training parameters
    Capabilities are data path, architecture, output directory, learning rate, hidden units, epochs, and gpu flag
    '''
    args = argparse.ArgumentParser()
    args.add_argument('data', help='Training data directory')
    args.add_argument('-a', '--arch', choices=['densenet121', 'vgg16'], required=True,  help='Architecture: densenet121 or vgg16')
    args.add_argument('-o', '--save_dir', required=True, help='Set directory to save checkpoints')
    args.add_argument('-l', '--learning_rate', required=False, help='Set hyperparameters: Learn Rate, defaults to 0.0025')
    args.add_argument('-hu', '--hidden_units', required=False, help='Set hyperparameters: Hidden Units')
    args.add_argument('-e', '--epochs', required=False, help='Set hyperparameters: Epochs, default to 1')
    args.add_argument('-g', '--gpu', required=False, help='Flag to use GPU for training')
    args.add_argument('-pe', '--print_every', required=False, help='Printout counter, every how many iterations to print out stats, default to 10')
    return args.parse_args()

def pred_args():
    '''
    Retrieves predicting parameters
    Capabilities are data path, architecture, output directory, learning rate, hidden units, epochs, and gpu flag
    '''
    args = argparse.ArgumentParser()
    args.add_argument('image', help='Image to predict')
    args.add_argument('checkpoint',  help='Model checkpoint with path included')
    args.add_argument('-g', '--gpu', help='Flag to use GPU for training')
    args.add_argument('-cn', '--category_names', required=False, help='Use Mapping of categories to real names')
    args.add_argument('-t', '--top_k', required=False, help='Return top K most likely classes')
    return args.parse_args()

    