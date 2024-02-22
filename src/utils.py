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
    args.add_argument('--data', help='Training data directory')
    args.add_argument('-a', '--arch', required=True,  help='Architecture: densenet121 or vgg16')
    args.add_argument('-o', '--save_dir', required=True, help='Set directory to save checkpoints')
    args.add_argument('-l', '--learning_rate', required=False, help='Set hyperparameters: Learn Rate')
    args.add_argument('-hu', '--hidden_units', required=False, help='Set hyperparameters: Hidden Units')
    args.add_argument('-e', '--epochs', required=False, help='Set hyperparameters: Epochs')
    args.add_argument('-d', '--gpu', required=False, help='Flag to use GPU for training')
    return args