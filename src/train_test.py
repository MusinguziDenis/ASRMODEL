import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import Levenshtein

# imports for decoding and distance calculation
import ctcdecode
from ctcdecode import CTCBeamDecoder

import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

def decode_prediction(output, output_lens, decoder, PHONEME_MAP):
    """"This fuction takes the output of the model and returns the decoded prediction
    Args:
        output: output of the model
        output_lens: lengths of the output
        decoder: decoder object
        PHONEME_MAP: phoneme map
    Returns:
        pred_strings: list of decoded predictions
    """
    
    (beam_results, beam_scores, timesteps, outlens) = decoder.decode(output.permute(1,0,2), seq_lens= output_lens) #lengths - list of lengths

    pred_strings                    = []
    
    # Loop through the batches in the input tensor and use the phoneme map to decode the predictions
    for i in range(output_lens.shape[0]):
        if outlens[i][0]!=0:
            curr_decode = "".join(PHONEME_MAP[j] for j in beam_results[i,0, :outlens[i][0]])
        pred_strings.append(curr_decode)
    return pred_strings

def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP): # y - sequence of integers
    """This function calculates the levenshtein distance between the output of the model and the labels
    Args:
        output: output of the model
        label: labels
        output_lens: lengths of the output
        label_lens: lengths of the labels
        decoder: decoder object
        PHONEME_MAP: phoneme map
    Returns:
        dist: levenshtein distance"""
    dist            = 0
    batch_size      = label.shape[0]
    # Call the decode_prediction function to get the decoded predictions
    pred_strings    = decode_prediction(output, output_lens, decoder, PHONEME_MAP)
    
    # Loop through the batches in the input tensor and use the phoneme map to decode the labels
    # and calculate the distance between the label and the prediction
    for i in range(batch_size):
        pred_string = [phoneme for phoneme in pred_strings[i]]
        
        label_string = [PHONEME_MAP[idx.to(torch.int64)] for idx in label[i]]
        
        dist += Levenshtein.distance(pred_string, label_string)

    dist /= batch_size
    
    return dist

def train_model(model, train_loader, criterion,scaler, optimizer, decoder, phoneme_map):
    """This function implements the train function of the model.
    Args:
        model: model object
        train_loader: train dataloader
        criterion: loss function
        scaler: scaler object
        optimizer: optimizer object
        decoder: decoder object
        phoneme_map: phoneme map
    Returns:
        total_loss: total loss
        vdist: levenshtein distance
    """
    model.train()# Ensure the model in train mode
    
    vdist= 0 # Initialize the levenshtein distance to 0
    
    # Initialize the tqdm progress bar
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    
    total_loss = 0 # Initialize the total loss to 0
    
    # Loop through the batches in the dataloader
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()# Zero the gradients
        
        x, y, lx, ly = data # Get the input and the labels from the batch
        
        x, y = x.to(device), y.to(device) # Move the input and the labels to the device
        # Apply autocast to use mixed precision in training
        with torch.cuda.amp.autocast():     
            
            h, lh = model(x, lx) # Get the output of the model
            
            h     = torch.permute(h, (1,0,2)) # Permute the output to match the input of the loss function 
            
            loss = criterion(h, y, lh, ly) # Calculate the loss

        # Extract the loss value from the loss tensor
        total_loss += loss.item()
        
        # Calculate the levenshtein distance
        vdist += calculate_levenshtein(h, y, lh, ly, decoder, phoneme_map)
        
        # Update the tqdm progress bar
        batch_bar.set_postfix(
            
            lev_dist = "{:0.04f}".format(float(vdist / (i + 1))),  
            
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update()

        # Another couple things you need for FP16. 
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        
        scaler.update() # This is something added just for FP16
        
        # delete the inputs and labels to free up memory
        del x, y, lx, ly, h, lh, loss 
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar
    
    return total_loss / len(train_loader), vdist/len(train_loader)


def validate_model(model, val_loader, decoder, phoneme_map, criterion):
    """"This function implements the validation function of the model.
    Params:
        model: model object
        val_loader: validation dataloader
        decoder: decoder object
        phoneme_map: phoneme map
        criterion: loss function"""
    model.eval() # Ensure that the model is in eval mode
    
    # Initialize the tqdm progress bar
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0 # Initialize the total loss to 0
    
    vdist = 0 # Initialize the levenshtein distance to 0
    
    # Loop through the batches in the dataloader
    for i, data in enumerate(val_loader):

        x, y, lx, ly = data # Get the input and the labels from the batch
        
        x, y = x.to(device), y.to(device) # Move the input and the labels to the device
        # Ensure that the model is in inference mode
        with torch.inference_mode():
        
            h, lh = model(x, lx) # Get the output of the model
        
            h = torch.permute(h, (1, 0, 2)) # Permute the output to match the input of the loss function
            loss = criterion(h, y, lh, ly) # Calculate the loss

        total_loss += float(loss) # Extract the loss value from the loss tensor
        
        vdist += calculate_levenshtein(h, y, lh, ly, decoder, phoneme_map) # Calculate the levenshtein distance
        
        # Update the tqdm progress bar
        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))
        batch_bar.update()
        
        # Delete the inpur and labels to free up memory
        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()
        
    batch_bar.close()# You need this to close the tqdm bar	
    
    total_loss = total_loss/len(val_loader) # Divide the total loss by the number of batches
    
    val_dist = vdist/len(val_loader) # Divide the total levenshtein distance by the number of batches
    
    return total_loss, val_dist

def save_model(model, optimizer, scheduler, metric, epoch, path):
    """This method saves a model checkpoints.
    Args:
        model: model object
        optimizer: optimizer object
        scheduler: scheduler object
        metric: metric value
        epoch: epoch number
        path: path to save the model
    """
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         metric[0]                  : metric[1], 
         'epoch'                    : epoch}, 
         path
    )
    
def load_model(path, model, metric= 'valid_acc', optimizer= None, scheduler= None):
    """This method loads a model checkpoints.
    Args:
        path: path to load the model
        model: model object
        metric: metric value
        optimizer: optimizer object
        scheduler: scheduler object
        Returns:
        -   model: the reloaded model
        -   optimizer: the reloaded optimizer
        -   scheduler: the reloaded scheduler
        -   epoch: the reloaded epoch
        -   metric: the reloaded metric"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    epoch   = checkpoint['epoch']
    metric  = checkpoint[metric]

    return [model, optimizer, scheduler, epoch, metric]