import argparse
import torch
from torch.autograd import Variable
import json
import numpy as np

default_top_k = 5

def load_model(filename = None):
    if filename != None:
        checkpoint = torch.load(filename)
        optimizer = checkpoint['optimizer']
        model = checkpoint['model']
        return model, optimizer

def load_cat_names(cat_to_name_filename):
    with open(cat_to_name_filename, 'r') as f:
        cat_to_name = json.load(f)

    print(cat_to_name)
    print("\n Lenght:", len(cat_to_name))
    return cat_to_name


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

    '''
    im = Image.open(image)

    im.resize(256, 256)

    value = 0.5 * (256 - 224)
    im = im.crop((value, value, 256 - value, 256 - value))

    im = np.array(image) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std

    image = image.transpose((2, 0, 1))

    return image

def make_prediction(image_path, model, topk = 5, enable_gpu = False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # move the model to cuda
    cuda = torch.cuda.is_available()
    if enable_gpu & cuda:
        # Move model parameters to the GPU
        model.cuda()
        print("Number of GPUs:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.device_count() - 1))
    else:
        model.cpu()
        print("We go for CPU")

    # turn off dropout
    model.eval()

    # The image
    image = process_image(image_path)

    # tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()

    # The image becomes the input
    image = Variable(image)
    if enable_gpu & cuda:
        image = image.cuda()

    output = model.forward(image)

    probabilities = torch.exp(output).data

    # getting the topk (=5) probabilites and indexes
    # 0 -> probabilities
    # 1 -> index
    prob = torch.topk(probabilities, topk)[0].tolist()[0]  # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0]  # index

    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return prob, label

def predict():
    parser = argparse.ArgumentParser()

    parser.add_argument('image')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names')
    parser.add_argument('--gpu', default=False, nargs='?')

    args = parser.parse_args()

    image_filename = args.image

    checkpoint_filename = args.checkpoint

    category_names_file = args.category_names

    top_k = args.top_k if args.top_k != None else default_top_k

    enable_gpu = args.gpu if args.gpu != None else False

    model, optimizer = load_model(checkpoint_filename)

    prop, label = make_prediction(image = image_filename, model = model, top_k = top_k, enable_gpu = enable_gpu)

    print(prop, label)

    if category_names_file != None:

        cat_to_name = load_cat_names(category_names_file)

        print(cat_to_name)

predict()






