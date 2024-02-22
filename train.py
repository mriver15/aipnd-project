from src import utils
# from src import dependencies



# @author mriver15 Michael Rivera
# Trains a network on a dataset and saves the model as a checkpoint.


def main():
    args = utils.train_args()
    print(args)

def train(model, dataloader, print_frequency, use_gpu):
    """
        trains model using dataloader train + validation profiles
        model object carries with it the criterion, optimizer, epochs
        print frequency goes by image processed, use_gpu determines device
    """

    # Device logic
    device = 'cpu'
    if use_gpu and torch.cude.is_available():
        mode.cuda()

    return -1



if __name__ == '__main__':
    main()
