import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import wandb
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from ctcdecode import CTCBeamDecoder
import warnings
warnings.filterwarnings('ignore')

from dataloader import AudioDataset, AudioDatasetTest
from model import ASRMODEL
from train_test import train_model, test_model, decode_prediction, save_model
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


PHONEMES = CMUdict[:-2]
LABELS = ARPAbet[:-2]

# Config file with hyperparameters and model parameters
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
train_dataset = AudioDataset(PHONEMES, partition='train-clean-100') #Load the training dataaet
dev_dataset = AudioDataset(PHONEMES, partition='dev-clean') # Load the validation dataset
test_dataset = AudioDatasetTest(PHONEMES, partition='test-clean') # Load the test dataset
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=dev_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=test_dataset.collate_fn)

# Create the model, optimizer, criterion, decoder, scheduler and scaler
model= ASRMODEL(input_channels=27,output_channels=256,embed_size=256, kernel_size=3, training=False)
model.to(device=device) # Move the model to the GPU
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
criterion = torch.nn.CTCLoss()
decoders   = CTCBeamDecoder(labels=LABELS, blank_id=0, beam_width=100, log_probs_input=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
scaler = torch.cuda.amp.GradScaler()


wandb.login(key='') # Enter wandb API key to login to wandb
run = wandb.init(project='ASR', 
                 reinit=True,
                #  id = '',
                #  resume='must',
                 project_name='ASR', 
                 config=config)

# Initialize the best validation loss to infinity
best_lev_dist = 'inf'
# Loop through the epochs
for epoch in config['epoch']:
    for param_group in optimizer.param_groups: # Loop through the parameters in the optimizer
        curr_lr = param_group['lr']
        print(f'Current learning rate: {curr_lr}')
    
    
    print("Start Train \t{}".format(epoch))
    startTime = time.time()
    train_loss = train_model(model, train_loader, criterion, scaler, optimizer)# Train the model
    print("Start Dev \t{}".format(epoch))
    valid_loss, valid_dist = test_model(model, dev_loader, criterion, decoders, PHONEMES, scaler) # Validate the model	
    print('***Saving Checkpoint ***')
    save_model(model, optimizer, epoch, train_loss, valid_loss, config['checkpoint_path']) # Save the model
    # Print the metrics
    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Dist {:.04f}%\t Val Loss {:.04f}".format(valid_dist, valid_loss))
    scheduler.step(valid_dist) # Step the scheduler	
    
    # Log metrics to Wandb
    wandb.log({
        'train_loss': train_loss,  
        'valid_dist': valid_dist, 
        'valid_loss': valid_loss, 
        'lr'        : curr_lr
    })
    # Save the model to wandb
    wandb.save(config['checkpoint_path'])
    
    # If the validation Levenshtein distance is less than the best validation Leveshtein distance, save the model
    if valid_dist <= best_lev_dist:
        best_valid_dist = valid_dist
        save_model(model, optimizer, epoch, train_loss, valid_loss, config['best_model_path'])
        wandb.save(config['best_model_path'])
        print('***Saving Best Model***')
        
run.finish() 
 
# Initialize the beam width for testing  
TEST_BEAM_WIDTH = config['test_beam_width']
results = [] # Initialize the results list

model.eval() # Ensure that the model is in eval mode
print("Start Test")
for data in tqdm(test_loader):# Loop through the batches in the test loader
    x, lx = data # Get the input and the input lengths from the batch
    x     = x.to(device) # Move the input to the GPU
    with torch.no_grad(): # Disable gradient calculation
        h, lh = model(x, lx) # Get the output of the model
    
    pred_strings = decode_prediction(h, lh, decoders, PHONEMES) # Decode the predictions
    results.extend(pred_strings) # Extend the results list with the decoded predictions
    # Delete the variables to free up memory
    del x, lx, h, lh
    torch.cuda.empty_cache()

# Create a dataframe with the predictions
df = pd.DataFrame(results, columns=['Predicted'])