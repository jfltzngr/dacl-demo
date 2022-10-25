import time, os, json, random 
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, F1Score

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish

from bikit.utils import list_datasets, download_dataset 
from bikit.datasets import BikitDataset 
from bikit.metrics import EMR_mt, Recalls_mt

from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Dictionary to find the suiting EfficientNet model according to the resolution of the input-images:
efnet_dict = {'b0': 224, 'b1': 240, 'b2': 260, 'b3': 300,   
              'b4': 380, 'b5': 456, 'b6': 528, 'b7': 600    
             }

class DaclNet(nn.Module):
    def __init__(self, base_name, resolution, hidden_layers, num_class, drop_prob=0.2, freeze_base=True):
        ''' 
        Builds a network separated into a base model and classifier with arbitrary hidden layers.
        
        Attributes
        ---------
        base_name:      string, basemodel for the NN
        resolution:     resolution of the input-images, example: 224, 240...(look efnet_dic), Only needed for EfficientNet
        hidden_layers:  list of integers, the sizes of the hidden layers
        drop_prob:      float, dropout probability
        freeze_base:    boolean, choose if you want to freeze the parameters of the base model
        num_class:      integer, size of the output layer according to the number of classes

        Example
        ---------
        model = Network(base_name='efficientnet', resolution=224, hidden_layers=[32,16], num_class=6, drop_prob=0.2, freeze_base=True)
        
        Note
        ---------
        -print(efficientnet) -> Last module: (_swish): MemoryEfficientSwish() and the last fc-layers are displayed
         This activation won't be called during forward due to: "self.base.extract_features"! No activation of last layer!
        '''
        super(DaclNet, self).__init__()
        # basemodel
        self.base_name = base_name
        self.resolution = resolution
        self.hidden_layers = hidden_layers
        self.freeze_base = freeze_base

        if self.base_name == 'mobilenet':
            base = models.mobilenet_v3_large(pretrained=True) 
            modules = list(base.children())[:-1] 
            self.base = nn.Sequential(*modules)
            # for pytorch model:
            if hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(base.classifier[0].in_features, self.hidden_layers[0])]) 
            else:
                self.classifier = nn.Linear(base.classifier[0].in_features, num_class)

            self.activation = nn.Hardswish()

        elif self.base_name == 'resnet':
            base = models.resnet50(pretrained=True) 
            modules = list(base.children())[:-1]
            self.base = nn.Sequential(*modules)
            if self.hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(base.fc.in_features, self.hidden_layers[0])])
            else:
                self.classifier = nn.Linear(base.fc.in_features, num_class)   
            self.activation = nn.ELU() 

        elif self.base_name == 'efficientnet':      
            for ver in efnet_dict:
                if efnet_dict[ver] == self.resolution:
                    self.version = ver
                    full_name = self.base_name+'-'+ver
            self.base = EfficientNet.from_pretrained(model_name=full_name) 
            if self.hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(self.base._fc.in_features, self.hidden_layers[0])])
            else:
                self.classifier = nn.Linear(self.base._fc.in_features, num_class)   
            self.activation = MemoryEfficientSwish()
            
        elif self.base_name == 'mobilenetv2':
            base = models.mobilenet.mobilenet_v2(pretrained=True)
            modules = list(base.children())[:-1]
            self.base = nn.Sequential(*modules)
            if hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(base.classifier[1].in_features, self.hidden_layers[0])]) 
            else:
                self.classifier = nn.Linear(base.classifier[1].in_features, num_class)
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError    
        
        # freeze the base
        if self.freeze_base:
            for param in self.base.parameters(): 
                param.requires_grad_(False)
        
        self.dropout = nn.Dropout(p=drop_prob, inplace=True)

        # classifier
        # Add a variable number of more hidden layers
        if self.hidden_layers:
            layer_sizes = zip(self.hidden_layers[:-1], self.hidden_layers[1:])        
            self.classifier.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
            # Add output layer to classifier
            self.classifier.append(nn.Linear(self.hidden_layers[-1], num_class))
        else:
            pass
        
    def forward(self, input_batch):
        ''' 
        Performs the feed-forward process for the input batch and returns the logits

        Arguments
        ---------
        input_batch: torch.Tensor, Multidimensional array holding elements of datatype: torch.float32, 
                     it's shape is: [1, 3, 224, 224] according to N x C x H x W,
                     The input batch carries all pixel values from the images inside teh batch
        Note
        ---------
        Every model uses 2d-Average-Pooling with output_size=1 after the feature extraction or rather before flattening.
        The pooling layer of ResNet50 and MobileNetV3 was kept in the squential -> Doesn't have to be called in forward!
        EffNet had to be implemented with the AdaptiveAvgpool2d in this forward function because of missing pooling when
        calling: "effnet.extract_features(input_batch)"
        Also mobilenetV2 needs the manually added pooling layer.

        Returns
        ---------
        logits: torch.Tensor, shape: [1, num_class], datatype of elements: float
        '''
        # Check if model is one that needs Pooling layer and/or special feature extraction
        if self.base_name in ['efficientnet', 'mobilenetv2']:
            if self.base_name == 'efficientnet':
                x = self.base.extract_features(input_batch)
            else:
                # For MobileNetV2
                x= self.base(input_batch)
            pool = nn.AdaptiveAvgPool2d(1)
            x = pool(x)
        else:
            # For any other model don't additionally apply pooling:
            x = self.base(input_batch)
        
        x = self.dropout(x)         # Originally only in EfficientNet a Dropout after feature extraction is added  
        x = x.view(x.size(0), -1)   # Or: x.flatten(start_dim=1)
        if self.hidden_layers:    
            for i,each in enumerate(self.classifier):
                # Put an activation function and dropout after each hidden layer
                if i < len(self.classifier)-1:
                    x = self.activation(each(x))
                    x = self.dropout(x)
                else:
                    # Don't use an activation and dropout for the last layer
                    logits = each(x)
                    break
        else:
            logits = self.classifier(x)

        return logits


def main():
    # In the cat_to_name file our damage-class-names are stored with the according position in the output vector:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)[cp['dataset']]

    # Choose which checkpoint/model you want to load from the table above:
    cp_name = 'Code_res_dacl.pth'

    # Load the checkpoint:
    cp = torch.load(Path('models/' + cp_name)) 

    # Instantiate the model:
    model = DaclNet(base_name=cp['base'], resolution = cp['resolution'], hidden_layers=cp['hidden_layers'], 
    				drop_prob=cp['drop_prob'], num_class=cp['num_class'])
    model.load_state_dict(cp['state_dict']) # Load the pre-trained weights into the model
    model.eval() # Set the model to eval-mode. No dropout and no autograd will be applied.
    model.to(device)

    # Define the test transforms:
    test_transforms = transforms.Compose([transforms.Resize(int(1.1*cp['resolution'])),
                                          transforms.CenterCrop(cp['resolution']),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Download dataset:
    download_dataset(cp['dataset']) 

    # Instantiate test-dataset and -loader from bikit:
    test_dataset = BikitDataset(name=cp['dataset'], split="test", transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print("======test_dataset======\n", test_dataset.df[test_dataset.class_names].sum())

    # Pack all metrics you want to calculate inside one MetricCollection from torchmetrics:
    metrics = MetricCollection([EMR_mt(use_logits=False),
                                F1Score(num_classes=test_dataset.num_classes, average='macro', compute_on_step=False),
                                Recalls_mt(num_classes=test_dataset.num_classes)]).to(device) # classwise Recall

    # Define your loss (Optional):
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    sum_counts = 0
    cumu_loss = 0

    start = time.time() # Save the starting time

    # Start the test loop:
    with torch.no_grad():
        for i, (data,targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            preds = torch.sigmoid(logits).float()
            loss = criterion(logits, targets)     
            bs = targets.shape[0]
            sum_counts += bs
            cumu_loss += loss.item() * bs

            metrics(preds, targets.int())

            cumu_preds = preds if i == 0 else torch.cat((cumu_preds, preds), 0)
            cumu_targets = targets if i == 0 else torch.cat((cumu_targets, targets), 0)

        total_loss = cumu_loss/sum_counts
        metrics = metrics.compute() # Compute the metrics after having finnished the test loop

    # Print all metrics you are curious about:    
    print('\n======Finnished Testing======')
    print('Tested dataset:    %s'      % cp['dataset'])
    print("Dacl's base-arch:  %s"      % cp['base'])
    print("TL-approach:       %s"      % cp['approach'])
    print('Test Loss:         %.4f'    % total_loss)
    print('ExactMatchRatio:   %.2f %%' % (metrics["EMR_mt"].item()*100))
    print('F1-Score:          %.2f'    % metrics["F1Score"].item())
    print('Time fore testing: %d s\n'  % (time.time() - start))
    for c in cat_to_name:
        print('Recall-%s: %.2f' % (cat_to_name[c], metrics['Recalls_mt'][int(c)]) )



    # Get the amount of completely correct predictions (Numerator in exact match ratio, EMR):
    correct = 0
    y_hat = (cumu_preds > .5) # Return True for each item in cumu_preds above threshold
    z_match = (y_hat == cumu_targets) # Return True for each item matching the corresponding one in the targets tensor
    z = torch.all(z_match, dim=1) # Check across the first dimension (width of the Tensor) if all items are True. If so return True for that sample.
    for i in z:
        if True in z[i]:
            # Count all exact matches:
            correct += 1
        else:
            pass
    print('\nCompletely correct predicted samples: %s' % correct) 
    print('Number of Samples in test dataset:    %s'   % len(test_dataset))

    # Now we can calculate the 'by hand'-EMR:
    byhand_emr = correct/cumu_targets.shape[0] 
    print("ExactMatchRatio calculated 'by hand': %.2f %%" % (byhand_emr * 100))

    y_hat_np = y_hat.to('cpu').numpy()
    y_hat_df = pd.DataFrame(y_hat_np)
    y_hat_df.to_csv('logs/y_hat_{}_{}.csv'.format(cp['dataset'], cp['base']))

    cumu_targets_np = cumu_targets.to('cpu').numpy()
    cumu_targets_df = pd.DataFrame(cumu_targets_np)
    cumu_targets_df.to_csv('logs/cumu_targets_{}_{}.csv'.format(cp['dataset'], cp['base']))

    z_np = z.to('cpu').numpy()
    z_df = pd.DataFrame(z_np)
    z_df.to_csv('logs/z_{}_{}.csv'.format(cp['dataset'], cp['base']))

    z_match_np = z_match.to('cpu').numpy()
    z_match_df = pd.DataFrame(z_match_np)
    z_match_df.to_csv('logs/z_match_{}_{}.csv'.format(cp['dataset'], cp['base']))

if __name__=="__main__":
    main()