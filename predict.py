import argparse
import torch
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import json
from network_prep import load_model
from processimage import process_image

def get_input_args():
    """
    Retrieves and parses command line arguments. This function returns these args
    as an ArgumentParser object.
        5 command line arguments are created:
            input - path to image file to predict
            checkpoint - path to checkpoint file
            top_k - number of predictions to output(default- 1)
            category_names - path to json file for mapped names(default- None)
            gpu - tells the program whether or not to select GPU processing(default - False/off)
        Parameters:
            None - use module to create & store command line arguments
        Returns:
            parse_args() - data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser(description='Get NN arguments')
    #Define arguments
    parser.add_argument('input', type=str, help='image to process and predict')
    parser.add_argument('checkpoint', type=str, help='cnn to load')
    parser.add_argument('--top_k', default=1, type=int, help='default top_k results')
    parser.add_argument('--category_names', default='', type=str, help='default category file' )
    parser.add_argument('--gpu', default=False, action='store_true', help='use GPU processing')
    
    return parser.parse_args()

def predict(image, model, top_k, gpu, category_names, arch, class_idx):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model. Returns top_k classes
    and probabilities. If name json file is passed, it will convert classes to actual names.
    '''
    image = image.unsqueeze(0).float()
    
    image = Variable(image, volatile=True)
    
    if gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
        print('GPU PROCESSING')
    else:
        print('CPU PROCESSING')
    
    out = model.forward(image)
    results = torch.exp(out).data.topk(top_k)
    classes = results[1][0]
    probs = Variable(results[0][0]).data
    
    #If category file path passed, convert classes to actual names
    if len(category_names) > 0:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        #Creates a dictionary of loaded names based on class_ids from model
        mapped_names = {}
        for k in class_idx:
            mapped_names[cat_to_name[k]] = class_idx[k]
        #invert dictionary to accept prediction class output
        mapped_names = {v:k for k,v in mapped_names.items()}
        
        classes = [mapped_names[x] for x in classes]
        probs = list(probs)
    else:
        #Invert class_idx from model to accept prediction output as key search
        class_idx = {v:k for k,v in class_idx.items()}
        classes = [class_idx[x] for x in classes]
        probs = list(probs)
    return classes, probs

def print_predict(classes, probs):
    """
    Prints predictions. Returns Nothing
    Parameters:
        classes - list of predicted classes
        probs - list of probabilities associated with class from classes with the same index
    Returns:
        None - Use module to print predictions
    """
    predictions = list(zip(classes, probs))
    for i in range(len(predictions)):
        print('{} : {:.3%}'.format(predictions[i][0], predictions[i][1]))
    pass

def main():
    in_args = get_input_args()
    norm_image = process_image(in_args.input)
    model, arch, class_idx = load_model(in_args.checkpoint)
    classes, probs = predict(norm_image, model, in_args.top_k, in_args.gpu, in_args.category_names, arch, class_idx)
    print_predict(classes, probs)
    pass
if __name__ == '__main__':
    main()