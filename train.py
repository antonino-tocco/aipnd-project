import matplotlib.pyplot as plt

import argparse
import os
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models

default_arch = "vgg16"
default_learning_rate = 0.01

input_nodes = {"vgg16": 25088, "densenet121": 1024, "alexnet": 9216}
default_hidden_units = 512
default_output_nodes = 102

default_epochs = 20

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    traindatasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validatasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    testdatasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloaders = torch.utils.data.DataLoader(traindatasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(validatasets, batch_size=64)
    testloaders = torch.utils.data.DataLoader(testdatasets, batch_size=64)

    return trainloaders, validloaders, testloaders


def get_model(arch, hidden_units, output_nodes):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_nodes[arch], hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, output_nodes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def train(data_dir=None, save_dir=None, arch="vgg16", learning_rate=0.01, hidden_units=512, epochs=20,
          enable_gpu=False):
    print("Begin train...")
    print_every = 20
    steps = 0
    training_loss = 0

    trainloaders, validloaders, testloaders = load_data(data_dir)
    model = get_model(arch, hidden_units, default_output_nodes)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    cuda = torch.cuda.is_available()
    if enable_gpu & cuda:
        model.cuda()
        print("Number of GPUs:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.device_count() - 1))
    else:
        model.cpu()
        print("We are on CPU")

    for e in range(epochs):

        model.train()

        for ii, (inputs, labels) in enumerate(trainloaders):
            steps += 1

            inputs = Variable(inputs)
            labels = Variable(labels)

            optimizer.zero_grad()

            if enable_gpu & cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            optimizer.step()

            training_loss += loss.data.item()

            if steps % print_every == 0:
                # Model in evaluation mode
                model.eval()  # turn off dropout

                # --------------------------------------------------
                # Start test loop
                accuracy = 0
                valid_loss = 0

                for jj, (inputs, labels) in enumerate(validloaders):

                    inputs = Variable(inputs, requires_grad=False)
                    labels = Variable(labels)

                    if cuda:
                        # Move input and label tensors to the GPU
                        inputs, labels = inputs.cuda(), labels.cuda()

                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.data.item()

                    ## Calculating the accuracy
                    # Model's output is log-softmax, take exponential to get the probabilities
                    ps = torch.exp(outputs).data

                    # Class with highest probability is our predicted class, compare with true label
                    # Gives index of the class with highest probability, max(ps)
                    equality = (labels.data == ps.max(1)[1])

                    # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                # End validation loop
                # --------------------------------------------------

                print("Epoch: {}/{}   ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(training_loss),
                      "Valid Loss: {:.3f}.. ".format(valid_loss),
                      "Valid Accuracy %: {:.3f}..".format(100 * accuracy / len(validloaders)))

                training_loss = 0

                # Model in training mode
                model.train()
    return model, optimizer


def save_checkpoint(model, optimizer, train_data, arch, hidden_units, lr):
    nn_filename = 'checkpoint.pth'
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'arch': arch,
        'hidden_units': hidden_units,
        'lr': lr
    }
    try:
        # os.remove(nn_path+nn_filename)
        os.remove(nn_filename)
        print("File is removed")
    except OSError:
        print("No such file")

    # Save checkpoint
    # torch.save(checkpoint, nn_path+nn_filename)
    torch.save(checkpoint, nn_filename)

    # Check if it is saved
    # if os.path.exists(nn_path+nn_filename):
    #    print("File is saved")
    #    print(os.listdir(nn_path))

    if os.path.exists(nn_filename):
        print("File is saved")
        print(os.listdir())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--arch')
    parser.add_argument('--learning_rate')
    parser.add_argument('--hidden_units')
    parser.add_argument('--epochs')
    parser.add_argument('--top_k')
    parser.add_argument('--gpu', default=True, nargs='?')

    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch if args.arch != None else default_arch
    learning_rate = float(args.learning_rate) if args.learning_rate != None else default_learning_rate
    hidden_units = args.hidden_units if args.hidden_units else default_hidden_units
    epochs = args.epochs if args.epochs != None else default_epochs
    enable_gpu = True if args.gpu != None else False

    print("data_dir ", data_dir)
    print("save_dir ", save_dir)
    print("arch ", arch)
    print("learning_rate ", learning_rate, type(learning_rate))
    print("hidden_units ", hidden_units)
    print("epochs ", epochs)

    model, optimizer, train_data = train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, enable_gpu)
    save_checkpoint(models, optimizer, train_data, arch, hidden_units, learning_rate)


main()