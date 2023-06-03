import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import Levenshtein
import wandb
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from ctcdecode import CTCBeamDecoder
import warnings
warnings.filterwarnings('ignore')

from ASRMODEL.dataloader import AudioDataset, AudioDatasetTest
from ASRMODEL.model import Network
from ASRMODEL.model import ASRMODEL
from ASRMODEL.train_test import train_model, test_model, decode_prediction, calculate_levenshtein, save_model, load_model
device = "cuda" if torch.cuda.is_available() else "cpu"

CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@", 
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W", 
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R", 
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w", 
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y", 
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D", 
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O", 
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())


PHONEMES = CMUdict[:-2]  #Changed this from [:-2]
LABELS = ARPAbet[:-2]


config = {
    'batch_size': 64,
    'lr': 1e-3,
    'epochs': 10,
    'checkpoint_path': '/home/ubuntu/ASRMODEL/checkpoint.pth',
    'best_model_path': '/home/ubuntu/ASRMODEL/best_model.pth',
    'optimizer': 'AdamW',
    'beam_width': 3,
    'test_beam_width': 10 
}

#Get dataloader 
train_dataset = AudioDataset(PHONEMES, partition='train-clean-100')
dev_dataset = AudioDataset(PHONEMES, partition='dev-clean')
test_dataset = AudioDatasetTest(PHONEMES, partition='test-clean')
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=dev_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=test_dataset.collate_fn)


model= ASRMODEL(input_channels=27,output_channels=256,embed_size=256, kernel_size=3, training=False)
model.to(device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
criterion = torch.nn.CTCLoss()
decoders   = CTCBeamDecoder(labels=LABELS, blank_id=0, beam_width=100, log_probs_input=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
scaler = torch.cuda.amp.GradScaler()


wandb.login(key='')
run = wandb.init(project='ASR', 
                 reinit=True,
                #  id = '',
                #  resume='must',
                 project_name='ASR', 
                 config=config)

best_lev_dist = 'inf'
for epoch in config['epoch']:
    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
        print(f'Current learning rate: {curr_lr}')
    
    
    print("Start Train \t{}".format(epoch))
    startTime = time.time()
    train_loss = train_model(model, train_loader, criterion, scaler, optimizer)
    print("Start Dev \t{}".format(epoch))
    valid_loss, valid_dist = test_model(model, dev_loader, criterion, decoders, PHONEMES, scaler)
    print('***Saving Checkpoint ***')
    save_model(model, optimizer, epoch, train_loss, valid_loss, config['checkpoint_path'])
    # Print the metrics
    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Dist {:.04f}%\t Val Loss {:.04f}".format(valid_dist, valid_loss))
    scheduler.step(valid_dist)
    
    # Log metrics to Wandb
    wandb.log({
        'train_loss': train_loss,  
        'valid_dist': valid_dist, 
        'valid_loss': valid_loss, 
        'lr'        : curr_lr
    })
    
    wandb.save(config['checkpoint_path'])
    
    
    if valid_dist <= best_lev_dist:
        best_valid_dist = valid_dist
        save_model(model, optimizer, epoch, train_loss, valid_loss, config['best_model_path'])
        wandb.save(config['best_model_path'])
        print('***Saving Best Model***')
        
run.finish()
    
TEST_BEAM_WIDTH = config['test_beam_width']
results = []

model.eval()
print("Start Test")
for data in tqdm(test_loader):
    x, lx = data
    x     = x.to(device)
    with torch.no_grad():
        h, lh = model(x, lx)
    
    pred_strings = decode_prediction(h, lh, decoders, PHONEMES)
    results.extend(pred_strings)
    
    del x, lx, h, lh
    torch.cuda.empty_cache()
    
df = pd.DataFrame(results, columns=['Predicted'])