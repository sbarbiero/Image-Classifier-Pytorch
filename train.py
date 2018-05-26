#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:29:53 2018

@author: sebastian
"""

import argparse
import torch
from torch.autograd import Variable
from network_prep import create_loaders, prep_model, create_classifier

def get_input_args():
    """
    Retrieves and parses command line arguments. This function returns these args
    as an ArgumentParser object.
        7 command line arguments are created:
            data_dir - Nonoptional, path to image files
            save_dir - Path to where checkpoint is stored(default- current path)
            arch - CNN model architecture to use for image classification(default- vgg
                   pick any of the following vgg, densenet, alexnet)
            learning_rate - Learning rate for the CNN(default - 0.001)
            hidden_units - sizes for hidden layers, expects comma seperated if more than one
            output_size - output size for data set training on(default-102)
            epochs - defines number of epochs to run(default- 10)
            gpu - turns gpu training on if selected(default- False/off)
        Parameters:
            None - use module to create & store command line arguments
        Returns:
            parse_args() - data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser(description='Get NN arguments')
    #Define arguments
    parser.add_argument('data_dir', type=str, help='mandatory data directory')
    parser.add_argument('--save_dir', default='', help='Directory to save checkpoint.')
    parser.add_argument('--arch', default='vgg', help='default architecture, options: vgg, densenet, resnet')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='default learning rate' )
    parser.add_argument('--hidden_units', default='512', type=str, help='default hidden layer sizes')
    parser.add_argument('--output_size', default=102, type=int, help='default output_size')
    parser.add_argument('--epochs', default=10, type=int, help='default training epochs')
    parser.add_argument('--gpu', default=False, action='store_true', help='use GPU processing')
    
    return parser.parse_args()

def train_classifier(model, trainloader, validloader, criterion, optimizer, epochs, gpu):
    """
    Trains the selected model based on parameters passed through command line arguments. Performs 
    validation loop every 40 steps and prints progress. This function returns the trained model. 
    Parameters:
        model - CNN architecture to be trained
        trainloader - pytorch data loader of training data
        validloader - pytorch data loader of data to be used to validate.
        criterion - loss function to be executed (default- nn.NLLLoss)
        optimizer - optimizer function to apply gradients (default- adam optimizer)
        epochs - number of epochs to train on
        gpu - boolean that flags GPU use
    Returns:
        model - Trained CNN
    """
    steps = 0
    print_every = 40
    run_loss = 0
    
    #Selects CUDA processing if gpu == True and if the environment supports CUDA
    if gpu and torch.cuda.is_available():
        print('GPU TRAINING')
        model.cuda()
    elif gpu and torch.cuda.is_available() == False:
        print('GPU processing selected but no NVIDIA drivers found... Training under CPU.')
    else:
        print('CPU TESTING')
        
    for e in range(epochs):
              
        model.train()
        
        #Training forward pass and backpropagation
        for images, labels in iter(trainloader):
            steps+= 1
            images, labels = Variable(images), Variable(labels)
            if gpu and torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()

            out = model.forward(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.data.item()
            
            #Runs validation forward pass and loop at specified interval
            if steps % print_every == 0:
                model.eval()

                acc = 0
                valid_loss = 0

                for images, labels in iter(validloader):
                    images, labels = Variable(images), Variable(labels)
                    if gpu and torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()
                    with torch.no_grad():
                        out = model.forward(images)
                        valid_loss += criterion(out, labels).data.item()
        
                        ps = torch.exp(out).data
                        equality = (labels.data == ps.max(1)[1])
        
                        acc += equality.type_as(torch.FloatTensor()).mean()
        
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                 "Training Loss: {:.3f}.. ".format(run_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}".format(acc/len(validloader)))  
    
                run_loss = 0
                model.train()
            
    print('{} EPOCHS COMPLETE. MODEL TRAINED.'.format(epochs))
    return model

def test_classifier(model, testloader, criterion, gpu):
    """
    Tests previously trained CNN on a test dataset and prints results. Returns nothing.
    Parameters:
        model - Trained CNN to test
        testloader - pytorch data loader of test data
        criterion - loss function to be executed (default- nn.NLLLoss)
        gpu - boolean that flags GPU use
    Returns:
        None - simply use test module to test the trained network
    """
    #Selects CUDA if gpu == True and environment supports CUDA
    if gpu and torch.cuda.is_available():
        print('GPU TESTING')
        model.cuda()
    elif gpu and torch.cuda.is_available() == False:
        print('GPU processing selected but no NVIDIA drivers found... testing under CPU.')
    else:
        print('CPU TESTING')
              
    model.eval()
              
    acc = 0
    test_loss = 0
    #Forward pass
    for images, labels in iter(testloader):
        images, labels = Variable(images), Variable(labels)
        if gpu and torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            out = model.forward(images)
            test_loss += criterion(out, labels).data.item()

            ps = torch.exp(out).data
            equality = (labels.data == ps.max(1)[1])

            acc += equality.type_as(torch.FloatTensor()).mean()

    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(acc/len(testloader)))
    pass

def save_model_checkpoint(model, input_size, epochs, save_dir, arch, learning_rate, class_idx, optimizer, output_size):
    """
    Saves the trained and tested module by outputting a checkpoint file.
    Parameters:
        model - Previously trained and tested CNN
        input_size - Input size used on the specific CNN
        epochs - Number of epochs used to train the CNN
        save_dir - Directory to save the checkpoint file(default- current path)
        arch - pass string value of architecture used for loading
    Returns:
        None - Use module to output checkpoint file to desired directory
    """
    saved_model = {
    'input_size':input_size,
    'epochs':epochs,
    'arch':arch,
    'hidden_units':[each.out_features for each in model.classifier if hasattr(each, 'out_features') == True],
    'output_size': output_size,
    'learning_rate': learning_rate,
    'class_to_idx': class_idx,
    'optimizer_dict': optimizer.state_dict(),
    'classifier': model.classifier,
    'state_dict': model.state_dict() 
    }
    #Save checkpoint in current directory unless otherwise specified by save_dir
    if len(save_dir) == 0:
        save_path = save_dir + 'checkpoint.pth'
    else:
        save_path = save_dir + '/checkpoint.pth'
    torch.save(saved_model, save_path)
    print('Model saved at {}'.format(save_path))
    pass

def main():
    in_args = get_input_args()
    trainloader, testloader, validloader, class_idx = create_loaders(in_args.data_dir)
    model, input_size = prep_model(in_args.arch)
    model, criterion, optimizer = create_classifier(model, input_size, in_args.hidden_units, in_args.output_size, in_args.learning_rate)
    trained_model = train_classifier(model, trainloader, validloader, criterion, optimizer, in_args.epochs, in_args.gpu)
    test_classifier(trained_model, testloader, criterion, in_args.gpu)
    save_model_checkpoint(trained_model, input_size, in_args.epochs, in_args.save_dir, in_args.arch, in_args.learning_rate, class_idx, optimizer, in_args.output_size)
    pass

if __name__ == '__main__':
    main()
