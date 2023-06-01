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
    
    (beam_results, beam_scores, timesteps, outlens) = decoder.decode(output.permute(1,0,2), seq_lens= output_lens) #lengths - list of lengths

    pred_strings                    = []
    
    for i in range(output_lens.shape[0]):
        if outlens[i][0]!=0:
            curr_decode = "".join(PHONEME_MAP[j] for j in beam_results[i,0, :outlens[i][0]])
        pred_strings.append(curr_decode)
    return pred_strings

def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP, decoders): # y - sequence of integers
    
    dist            = 0
    batch_size      = label.shape[0]
    pred_strings    = decode_prediction(output, output_lens, decoders, PHONEME_MAP)
    
    for i in range(batch_size):
        pred_string = [phoneme for phoneme in pred_strings[i]]
        label_string = [PHONEME_MAP[idx.to(torch.int64)] for idx in label[i]]
        dist += Levenshtein.distance(pred_string, label_string)

    dist /= batch_size
    return dist

def train_model(model, train_loader, criterion,scaler, optimizer):
    
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():     
            h, lh = model(x, lx)
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update()

        # Another couple things you need for FP16. 
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        del x, y, lx, ly, h, lh, loss 
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar
    
    return total_loss / len(train_loader)


def validate_model(model, val_loader, decoder, phoneme_map, criterion):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            h, lh = model(x, lx)
            # h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += float(loss)
        vdist += calculate_levenshtein(torch.permute(h, (0, 1, 2)), y, lh, ly, decoder, phoneme_map)#changed the permute (1,0,2)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))
        batch_bar.update()
    
        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()
        
    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    val_dist = vdist/len(val_loader)
    
    return total_loss, val_dist

def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         metric[0]                  : metric[1], 
         'epoch'                    : epoch}, 
         path
    )
    
def load_model(path, model, metric= 'valid_acc', optimizer= None, scheduler= None):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    epoch   = checkpoint['epoch']
    metric  = checkpoint[metric]

    return [model, optimizer, scheduler, epoch, metric]