import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class PermuteBlock(nn.Module):
    def forward(self, x):
        return x.transpose(1,2)
    
    
class pBLSTM(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        self.blstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)# TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size

    def forward(self, x_packed):

        x, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        
        x, x_lens = self.trunc_reshape(x, x_lens)
        packed_in = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        packed_out, packed_length = self.blstm(packed_in)

        return packed_out

    def trunc_reshape(self, x, x_lens): 
        if x.size(1) % 2 == 1:
            x = F.pad(x, (0,0,0,1,0,0), "constant", 0)
        x = torch.reshape(x, (x.size(0), x.size(1)//2, x.size(2)*2))
        x_lens  = torch.clamp(x_lens, max=x.shape[1])
        return x, x_lens
    

class Network(nn.Module):
    
    def __init__(self, input_channels, ouput_channels, hidden_size, kernel_size, outsize, padding, stride):
        
        super(Network, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = ouput_channels
        self.hidden_size     = hidden_size
        self.kernel_size     = kernel_size
        self.out_size        = outsize
        self.padding         =  padding
        self.stride          = stride
        
        self.embedding       =  nn.Sequential(
            nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, self.padding, self.stride),
            nn.BatchNorm1d(self.output_channels),
            nn.GELU()
        )
        
        self.lstm             = nn.LSTM(self.output_channels, self.hidden_size, num_layers=2, bidirectional =True, bias =True, batch_first =True, dropout=0.2)
        self.classification   = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.out_size)
        )
        
        self.logSoftmax        = nn.LogSoftmax(dim=2)
        self.permutate         = PermuteBlock()
        
    def forward(self, x, lx):
        x_cnn                       = self.permutate(x)  # (B, T, C) -> (B, C, T) Reorder the channels and time channel input to CNN
        y_cnn                       = self.embedding(x_cnn) # Conv1d return (B,C, T)
        x_lstm                      = self.permutate(y_cnn) # Permutate the channels back to (B, T, C)
        x_ltsm_input                = pack_padded_sequence(x_lstm, lx, enforce_sorted=False, batch_first=True)
        y_ltsm, (h_n, c_n)          = self.lstm(x_ltsm_input)
        unpacked, unpacked_length   = pad_packed_sequence(y_ltsm, batch_first=True)
        logits                      = self.classification(unpacked)
        output_prob                 = self.logSoftmax(logits)
        ctcloss_input               = output_prob.permutate(1, 0, 2)
        
        return ctcloss_input, lx
    

class Encoder(nn.Module):
    def __init__(self, input_channels, ouput_channels, kernel_size, padding, stride, encoder_hidden_size) -> None:
        super(Encoder, self).__init__()
        
        self.input_channels         = input_channels
        self.output_channels        = ouput_channels
        self.encoder_hidden_size    = encoder_hidden_size
        self.kernel_size            = kernel_size
        self.padding                = padding
        self.stride                 = stride
        self.permutate              = PermuteBlock()
        
        self.embedding              = nn.Sequential(
            nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, self.padding, self.stride),
            nn.BatchNorm1d(self.output_channels),
            nn.GELU()
        )
        
        self.pBLSTMS    	        = torch.nn.Sequential(
                                        pBLSTM(encoder_hidden_size*4, encoder_hidden_size),
                                        pBLSTM(encoder_hidden_size*4, encoder_hidden_size),
                                        pBLSTM(encoder_hidden_size*4, encoder_hidden_size)
        )
        
    def forward(self, x, xlen):
        x_cnn                           = self.permutate(x)
        y_cnn                           = self.embedding(x_cnn)
        x_lstm                          = self.permutate(y_cnn)
        packed_input                    = pack_padded_sequence(x_lstm, xlen, enforce_sorted=False, batch_first=True)
        pblstm_output, _                = self.pBLSTMS(packed_input)
        encoder_outputs, encoder_lens   = pad_packed_sequence(pblstm_output, batch_first=True)
        
        return encoder_outputs, encoder_lens
    
    
class Decoder(nn.Module):
    def __init__(self, embed_size, output_size =41) -> None:
        super(Decoder, self).__init__()
        
        self.mlp            = nn.Sequential(
            PermuteBlock(), nn.BatchNorm1d(embed_size), PermuteBlock(),
            nn.Linear(embed_size, embed_size *4),
            nn.BatchNorm1d(embed_size *4),
            nn.Dropout(p = 0.2),
            torch.nn.Tanh(),
            nn.Linear(embed_size* 4, output_size)
        ),
        
        self.softmax         = nn.LogSoftmax(dim=2)
        
    def forward(self, encoder_out):
        out = self.softmax(self.mlp(encoder_out))
        
        return out
    
    
class ASRMODEL(nn.Module):
    def __init__(self, input_size,input_channels, kernel_size,  output_channels,  embed_size= 192, output_size= 41, training=True) -> None:
        super(ASRMODEL, self).__init__()
        
        self.training                   = training
        
        self.augmentations              = nn.Sequential(
            
        )
        
        self.encoder                    = Encoder(input_channels, output_channels, kernel_size, padding=1, stride=1, encoder_hidden_size=embed_size)
        self.decoder                    = Decoder(embed_size,output_size)
        
    def forward(self,x, lengths_x):
        if self.training:
            x                           = self.augmentations(x)
            
        encoder_out, encoder_len        = self.encoder(x, lengths_x)
        decoder_out                     = self.decoder(encoder_out)
        
        return decoder_out, encoder_len