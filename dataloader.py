import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


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

root ='/home/ubuntu'


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, PHONEMES, context =0, partition ='train-clean-100'): 
        '''
        '''
        self.mfcc_dir = os.path.join(root,partition, 'mfcc')
        self.transcript_dir = os.path.join(root, partition, 'transcript')
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        self.transcript_files = sorted(os.listdir(self.transcript_dir))
        self.PHONEMES = PHONEMES        
        # Making sure that we have the same no. of mfcc and transcripts
        assert len(self.mfcc_files) == len(self.transcript_files)      
        self.length = len(self.mfcc_files)
        self.mfccs = list()
        self.transcripts = list()

        for i in range(self.length):
            mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i]))
            mfcc = (mfcc - np.mean(mfcc, axis =0))/np.std(mfcc, axis =0)
            transcript = np.load(os.path.join(self.transcript_dir, self.transcript_files[i]))
            y = np.array([PHONEMES.index(i) for i in transcript[1:-1]])
            self.mfccs.append(mfcc)
            self.transcripts.append(y)

    def __len__(self):
        return self.length
    
    def __getitem__(self, ind):
        '''
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

    def __init__(self, PHONEMES, context =0, partition ='test-clean'): 
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''
        self.mfcc_dir = os.path.join(root, partition, 'mfcc') 
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        self.PHONEMES = PHONEMES        
        self.length = len(self.mfcc_files)
        self.mfccs = list()
  
        for i in range(self.length):
            mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i]))
            mfcc = (mfcc - np.mean(mfcc, axis =0))/np.std(mfcc, axis =0)
            self.mfccs.append(mfcc)
    
    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        '''
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