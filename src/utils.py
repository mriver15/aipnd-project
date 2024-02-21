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
    args.add_argument('-d', '--data', help='Training data directory')
    args.add_argument('-a', '--arch', required=True,  help='Architecture: densenet121 or vgg16')
    args.add_argument('-o', '--output-dir', required=True, help='Output directory')
    args.add_argument('-l', '--learn-rate', required=False, help='Learn Rate - .0025 by default')
    args.add_argument('-h', '--hidden-units', required=False, help='Set number of hidden units')
    args.add_argument('-e', '--epochs', required=False, help='Set training epochs')
    args.add_argument('-d', '--data', required=False, help='Training data directory')
