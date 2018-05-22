import torch
import sys
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict

def create_loaders(data_dir):
    """
    Creates pytorch training, validation and testing pytorch dataloaders and applies transformations
    respectively.
    Parameters:
        data_dir - Path to data to be used
    Returns:
        trainloader - Normalized training data loader with random crops, flipping and resizing applied
        testloader - Normalized testing data loader with fixed cropping and resizing
        validloader - Normalized validation data loader with fixed cropping and resizing
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    #Define data transforms
    data_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(norm_mean,
                                                             norm_std)])
    #Define training data transforms
    data_trans_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_mean,
                                                          norm_std)])
    #Load the datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=data_trans_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms)
    
    #Define dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    
    #Grab class ids
    class_idx = train_dataset.class_to_idx
    
    return trainloader, testloader, validloader, class_idx

def prep_model(arch):
    """
    Selects, downloads and returns model provided. Returns model architecture for the CNN and
    the associated input_size.
    Parameters:
        arch - Used to select which architecture to use for prepare
    Returns:
        model_select[arch] - selects the variable out of a dictionary and returns the
            model associated with arch
        input_size[arch] - selects the associated input size for the model selected
            with arch
    """
    vgg16=''
    alexnet=''
    densenet121=''
    #Only download the model you need, kill program if one of the three models isn't passed
    if arch == 'vgg':
        vgg16 = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
    elif arch == 'densenet':
        densenet121 = models.densenet121(pretrained=True)
    else:
        print('{} architecture not recognized. Supported args: \'vgg\', \'alexnet\', or \'densenet\''.format(arch))
        sys.exit()
    
    #Pass the model, and grab the input size
    model_select = {'vgg':vgg16,'alexnet':alexnet,'densenet':densenet121}
    input_size = {'vgg':25088,'alexnet':9216,'densenet':1024}
    return model_select[arch], input_size[arch]

def create_classifier(model, input_size, hidden_layers, learning_rate, drop_p=0.5):
    """
    Takes a pretrained CNN, freezes the features and creates a untrained classifier. Returns
    model with an untrained classifier, loss function critierion (NLLLoss) and Adam optimizer.
    Parameters:
        model - Pretrained CNN
        input_size - determines the size of the first input layer
        hidden_layers - comma separated string out_features for each layer including output size
            (requires 3 parameters)
        learning_rate - determines the learning rate for the optimizer
        drop_p - determines the dropout probability for the classifier(default- 0.5)
    Returns:
        model - Pretrained feature CNN with untrained classifier
        criterion - loss function to train on (torch.nn.NLLLoss())
        optimizer - optimizer for new, untrained classifier (torch.optim.Adam)
    """
    #Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
    
    hidden_layers = hidden_layers.split(',')
    hidden_layers = [int(x) for x in hidden_layers]
    
    if not len(hidden_layers) == 3:
        print('Error: Please pass 3 comma seperated values into hidden_units argument.')
        print('See module definitions for help.')
        sys.exit()
    #Take hidden_layer sizes and creates layer size definitions for each hidden_layer size combo
    layers = nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
    layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
    layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    
    #Define classifier
    classifier = nn.Sequential(OrderedDict([
        ('drop1', nn.Dropout(p=drop_p)),
        ('fc1', layers[0]),
        ('rel_1', nn.ReLU()),
        ('drop2', nn.Dropout(p=drop_p)),
        ('fc2', layers[1]),
        ('rel_2', nn.ReLU()),
        ('drop3', nn.Dropout(p=drop_p)),
        ('fc3', layers[2]),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    #Apply new classifier and generate criterion and optimizer
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer

def load_model(checkpoint):
    trained_model = torch.load('checkpoint.pth')
    arch = trained_model['arch']
    class_idx = trained_model['class_to_idx']
    #Only download the model you need, kill program if one of the three models isn't passed
    if arch == 'vgg':
        load_model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        load_model = models.alexnet(pretrained=True)
    elif arch == 'densenet':
        load_model = models.densenet121(pretrained=True)
    else:
        print('{} architecture not recognized. Supported args: \'vgg\', \'alexnet\', or \'densenet\''.format(arch))
        sys.exit()
        
    for param in load_model.parameters():
        param.requires_grad = False
    
    load_model.classifier = trained_model['classifier']
    load_model.load_state_dict(trained_model['state_dict'])
    
    return load_model, arch, class_idx

if __name__ == '__main__':
    print('This is run as main.')