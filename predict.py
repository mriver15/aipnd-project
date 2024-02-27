# @author mriver15 Michael Rivera
# Uses a trained network to predict the class for an input image
# python predict.py flowers/test/1/image_06743.jpg runs/model_20240227_1601_1 --gpu cpu --category_names cat_to_name.json --top_k 10
from src import utils
from torchvision import transforms, models
import torch
from PIL import Image
import math

import json

def load_model(path):
    """
    Returns model from checkpoint
    @param path path to model checkpoint
    """
    checkpoint = torch.load(path)

    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)

    model.classifier = checkpoint['classifier']

    print(checkpoint['optimizer_state_dict'])

    return model


def process_img(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

        @param image image path to process
    '''
    shortest_side = 256
    crop_size = 224
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    with Image.open(image) as im:
        # Get thumbnail
        if min(im.width, im.height) == im.height:
                # height is smallest side
            im = im.resize((math.floor((im.width * shortest_side)/im.height), shortest_side))
        else:
            # Width is smallest side or equal
            im = im.resize((shortest_side,math.floor((im.height * shortest_side)/im.width)))

        # Define Crop Sizing
        # Should be adaptable to 224 x 224 if sizes are larger than 256 
        l = (im.width - crop_size)/2
        t = (im.height - crop_size)/2
        r = crop_size + l
        b = crop_size + t
        # Crop Image
        im = im.crop((l, t, r, b))

        # Normalizing
        im_transform = transforms.ToTensor()
        im_norm = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        im_tensor = im_norm(im_transform(im))

        # im_prep = np.array(im_tensor)

        # Uncomment next to show image
        # imshow(im_tensor)
        return im_tensor # TODO: Check if this is the correct data to send back

def predict_to_string(ps_list, id_list, labels):
    '''
    Structures two lists into a small table displaying prediction results
    '''
    print("Prediction Matrix - Top {} Results".format(len(ps_list)))
    print("---------------------------------")
    for i,r in enumerate(range(len(ps_list))):
        idl = id_list[i]
        print("{:.2f}% | {} ..".format(ps_list[i]*100, labels[int(idl)]))

def predict(image, model, topk=5):
    """
    Predicts image category using model returns topk results
    @param image image loaded and preprocessed
    @param model model loaded from checkpoint
    @param topk top results
    """
    pre_im = torch.unsqueeze(process_img(image),0)

    model.eval()

    log_ps = model.forward(pre_im)
    ps = torch.exp(log_ps)
    top_ps, top_idx = ps.topk(topk, dim=1)

    ps_list = top_ps.tolist()[0]
    id_list = top_idx.tolist()[0]

    return ps_list, [str(a) for a in id_list]



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

    model = load_model(args.checkpoint)
    img = args.image

    if not args.gpu:
        device = 'cpu'
    else:
        device = 'gpu'

    if args.category_names:
        labels = get_category_mapping(args.category_names)
    else:
        labels = None

    if args.top_k:
        topk = args.top_k
    else:
        topk = None
    
    pss, ids = predict(img, model, topk)

    predict_to_string(pss,ids,labels)


if __name__ == '__main__':
    main()
