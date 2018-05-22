import torch
from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        Parameters:
            image - path to image to process
        Returns:
            torch.from_numpy(new_image) - torch DoubleTensor from process Numpy Array
    '''
    image_raw = Image.open(image)
    image_raw.thumbnail((256,256), Image.ANTIALIAS)
    width, height = image_raw.size
    crop_width = 224
    crop_height = 224
    
    #Calc the center boundaries
    left = (width - crop_width)/2
    upper = (height - crop_height)/2
    right = (width + crop_width)/2
    lower = (height + crop_height)/2
    
    #Crop image
    image_raw = image_raw.crop((left, upper, right, lower))
    
    #Establish Normalization Values
    norm_mean = np.array([0.485,0.456,0.406])
    norm_std = np.array([0.229, 0.224, 0.225])
    
    #Normalize
    np_image_raw = np.array(image_raw, dtype=np.float64)
    np_image_raw = np_image_raw / 255.0
    np_image_raw = (np_image_raw - norm_mean) / norm_std
    
    new_image = np_image_raw.transpose(2,0,1)
    
    return torch.from_numpy(new_image)