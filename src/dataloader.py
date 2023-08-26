import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import CMUdict_ARPAbet, CMUdict, ARPAbet, PHONEMES, LABELS

root ='/home/ubuntu'# Root folder with the dataset

class AudioDataset(torch.utils.data.Dataset):
    """Implements a custom dataloader class for the training and validation data.
    Parameters
    ----------
    PHONEMES : list
        List of phonemes
    partition : str
        Dataset partition to load
    """
    def __init__(self, PHONEMES, partition ='train-clean-100'): 
        
        self.mfcc_dir = os.path.join(root,partition, 'mfcc') # Create a path for loading the mfcc files
        
        self.transcript_dir = os.path.join(root, partition, 'transcript') # Create a path for loading the transcript files
        
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)) # Get the list of sorted mfcc files
        
        self.transcript_files = sorted(os.listdir(self.transcript_dir))# Get the list of sorted transcript files
        
        self.PHONEMES = PHONEMES       
        # Making sure that we have the same no. of mfcc and transcripts
        assert len(self.mfcc_files) == len(self.transcript_files)      
        
        self.length = len(self.mfcc_files) # Set the length of the dataset
        
        self.mfccs = list() # Create a list of loaded mfccs
        
        self.transcripts = list() # Create a list of loaded transcripts
        # Loop through the files and load the mfccs and transcripts
        for i in range(self.length):
            
            mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i])) # Load the mfcc file
            
            mfcc = (mfcc - np.mean(mfcc, axis =0))/np.std(mfcc, axis =0) # Normalize the mfcc coefficients
            
            transcript = np.load(os.path.join(self.transcript_dir, self.transcript_files[i])) # Load the transcript file
            
            y = np.array([PHONEMES.index(i) for i in transcript[1:-1]]) # Convert the transcript to a list of phonemes
            
            self.mfccs.append(mfcc) # Append the mfcc to the list of mfccs
            
            self.transcripts.append(y) # Append the transcript to the list of transcripts

    def __len__(self):
        """Length of the dataset"""
        return self.length
    
    def __getitem__(self, ind):
        '''
        Load a single example from the dataset
        args:
            ind: index of the example to be loaded
        returns:
            mfcc: mfcc coefficients
            transcript: transcript
        '''
        
        mfcc =  torch.FloatTensor(self.mfccs[ind])
        
        transcript = torch.FloatTensor(self.transcripts[ind])
        
        return mfcc, transcript

    def collate_fn(self,batch):
        '''
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish. 
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features, 
            and lengths of labels.
        '''
        # batch of input mfcc coefficients
        batch_mfcc, batch_transcript = zip(*batch)
        
        # batch of output phonemes
        batch_mfcc_pad =  pad_sequence(batch_mfcc,batch_first =True)
        
        lengths_mfcc = [element.size(dim=0) for element in batch_mfcc] 
        
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)
        
        lengths_transcript = [element.size(dim=0) for element in batch_transcript]
        
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)


class AudioDatasetTest(torch.utils.data.Dataset):
    """Implements a custom dataloader class for the test data.
    Parameters
    ----------
    PHONEMES : list
        List of phonemes
    partition : str
        Dataset partition to load
    """
    def __init__(self, PHONEMES, partition ='test-clean'): 
        
        self.mfcc_dir = os.path.join(root, partition, 'mfcc') # Create pats for loading the mfcc files
        
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)) # Get the list of sorted mfcc files
        
        self.PHONEMES = PHONEMES # Get the list of phonemes
        
        self.length = len(self.mfcc_files) # Set the length of the dataset
        
        self.mfccs = list() # Create a list of loaded mfccs
        # Loop through the files and load the mfccs
        for i in range(self.length):
        
            mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i])) # Load the mfcc file
        
            mfcc = (mfcc - np.mean(mfcc, axis =0))/np.std(mfcc, axis =0) # Normalize the mfcc coefficients
        
            self.mfccs.append(mfcc) # Append the mfcc to the list of mfccs
    
    def __len__(self):
        """"Returns the length of the dataset"""
        return self.length

    def __getitem__(self, ind):
        '''
        Load a single example from the dataset
        args:
            ind: index of the example to be loaded
        returns:
            mfcc: mfcc coefficients
        '''
        mfcc =  torch.FloatTensor(self.mfccs[ind])
        return mfcc

    def collate_fn(self,batch):
        '''
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish. 
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features, 
            and lengths of labels.
        '''
        # batch of input mfcc coefficients
        batch_mfcc = [data for data in batch]    
        
        batch_mfcc_pad =  pad_sequence(batch_mfcc,batch_first =True)
        
        lengths_mfcc = [element.size(dim=0) for element in batch_mfcc] 
        
        return batch_mfcc_pad, torch.tensor(lengths_mfcc)