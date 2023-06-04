import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torchaudio

class PermuteBlock(nn.Module):
    """Implemetation of a permutation function to be used in the model especially for the CNN part"""
    def forward(self, x):
        return x.transpose(1,2)
    
    
class pBLSTM(torch.nn.Module):
    """Implemetation of a pBLSTM layer. This layer is used to reduce the time dimension of the input by a factor of 2
    and increases the channel dimension of the input by a factor of 2. It is meant to allow the attention module to extract information 
    from a shorter sequence. It also allows the model to learn nonlinear feature representations of the data.
    It also reduced the computation complexity of the model."""	
    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        self.blstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)# TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size

    def forward(self, x_packed):

        x, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        
        x, x_lens = self.trunc_reshape(x, x_lens)
        packed_in = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted = False)
        packed_out, packed_length = self.blstm(packed_in)

        return packed_out

    def trunc_reshape(self, x, x_lens): 
        if x.size(1) % 2 == 1:# Applying padding for sequences with odd length
            x = F.pad(x, (0,0,0,1,0,0), "constant", 0)
        x = torch.reshape(x, (x.size(0), x.size(1)//2, x.size(2)*2))
        x_lens  = torch.clamp(x_lens, max=x.shape[1])
        return x, x_lens
    

class Network(nn.Module):
    """Implements a simple network with a CNN and a LSTM. The CNN is used to extract features from the input and the LSTM is used to encode the input sequence.
    
    Parameters
    ----------
    input_channels : int
        Number of channels in the input mfcc features
    ouput_channels : int
        Number of channels in the output of the CNN
    hidden_size : int
        Number of hidden units in the LSTM
    kernel_size : int
        Size of the kernel in the CNN
    outsize : int
        Number of output classes
    padding : int
        Padding size in the CNN
    stride : int
        Stride size in the CNN
    """
    
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
        """This function takes the input sequence of mfcc features encodes them using a CNN and a LSTM and then classifies them using an MLP.
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of mfcc features of shape (B, T, C)
        lx : torch.Tensor
            Lengths of the input sequences of shape (B)
        Returns
        -------
        ctcloss_input : torch.Tensor
            Output of the log softmax layer of shape (T, B, C)
        lx : torch.Tensor
            Lengths of the output sequences of shape (B)
        """
        x_cnn                       = self.permutate(x)  # (B, T, C) -> (B, C, T) Reorder the channels and time channel input to CNN
        y_cnn                       = self.embedding(x_cnn) # Conv1d return (B,C, T)
        x_lstm                      = self.permutate(y_cnn) # Permutate the channels back to (B, T, C)
        x_ltsm_input                = pack_padded_sequence(x_lstm, lx, enforce_sorted=False, batch_first=True)#Ensure that the sequences have the same length
        y_ltsm, (h_n, c_n)          = self.lstm(x_ltsm_input)# (B, T, C) -> (B, T, C*2) # TODO: Apply the LSTM to x_lstm, also you need to use pack_padded_sequence before you apply the LSTM and then pad_packed_sequence after you apply the LSTM
        unpacked, unpacked_length   = pad_packed_sequence(y_ltsm, batch_first=True)# Unpack the sequence after the LSTM layer
        logits                      = self.classification(unpacked)# Pass the extracted features through an MLP to get the logits
        output_prob                 = self.logSoftmax(logits)# Pass the logits through a log softmax layer to get the log probabilities
        ctcloss_input               = output_prob.permutate(1, 0, 2)# Permutate the output of the log softmax layer to be in the form expected by the ctc loss function
        
        return ctcloss_input, lx
    

class Encoder(nn.Module):
    """Implements the encoder of the ASR model. The encoder is composed of a CNN and a LSTM. The CNN is used to extract features from the input and the LSTM is used to encode the input sequence.
    Parameters
    ----------
    input_channels : int
        Number of channels in the input mfcc features
    ouput_channels : int
        Number of channels in the output of the CNN
    kernel_size : int
        Size of the kernel in the CNN
    padding : int
        Padding size in the CNN
    stride : int
        Stride size in the CNN
    encoder_hidden_size : int
        Number of hidden units in the LSTM
    permutate : PermuteBlock
        Permutation function to be used in the model to reorder the channels and time channel input to CNN
    """
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
        self.lstm                   = nn.LSTM(input_channels, encoder_hidden_size, batch_first =True, num_layers=2, dropout= 0.2, bidirectional=True)
        
        self.pBLSTMS    	        = torch.nn.Sequential(
                                        pBLSTM(encoder_hidden_size*4, encoder_hidden_size),
                                        pBLSTM(encoder_hidden_size*4, encoder_hidden_size),
                                        pBLSTM(encoder_hidden_size*4, encoder_hidden_size)
        )
        
    def forward(self, x, xlen):
        """This function takes input sequence of mfccs and embeds them using a LSTM and a pBLSTM.
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of mfcc features of shape (B, T, C)
        xlen : torch.Tensor
            Lengths of the input sequences of shape (B)
        Returns
        -------
        encoder_outputs : torch.Tensor
            Output of the encoder of shape (B, T, C)
        encoder_lens : torch.Tensor
            Lengths of the output sequences of shape (B)
        """
        packed_input                    = pack_padded_sequence(x, xlen, enforce_sorted=False, batch_first=True)#Ensure that the features have the same length before passing them
        x_lstm,_                        = self.lstm(packed_input)
        pblstm_output                   = self.pBLSTMS(x_lstm)
        encoder_outputs, encoder_lens   = pad_packed_sequence(pblstm_output, batch_first=True)
        
        return encoder_outputs, encoder_lens
    
    
class Decoder(nn.Module):
    """This class iplements the decoder of the ASR model. The decoder is a simple MLP that takes the output of the encoder and classifies it into the different phonemes.
    Parameters
    ----------
    embed_size : int
        Number of hidden units in the MLP
    output_size : int
        Number of output classes
    """
    def __init__(self, embed_size, output_size =41) -> None:
        super(Decoder, self).__init__()
        
        self.mlp            = nn.Sequential(
            nn.Linear(embed_size, embed_size *4),
            PermuteBlock(), nn.BatchNorm1d(embed_size *4), PermuteBlock(),
            nn.Dropout(p = 0.2),
            torch.nn.Tanh(),
            nn.Linear(embed_size* 4, output_size)
        )
        
        self.softmax         = nn.LogSoftmax(dim=2)
        
    def forward(self, encoder_out):
        """Implements the forward pass of the decoder. This takes the outputs of the encoder and classifies them into the different phonemes using an MLP network.
        Parameters
        ----------
        encoder_out : torch.Tensor
            Output of the encoder of shape (B, T, C)
        """
        x   = self.mlp(encoder_out)
        out = self.softmax(x)
        
        return out
    
    
class ASRMODEL(nn.Module):
    """This implements the ASR model. The model is composed of an encoder and a decoder. The encoder is composed of a CNN and a LSTM. 
    The CNN is used to extract features from the input and the LSTM is used to encode the input sequence.
    Parameters
    ----------
    input_channels : int
        Number of channels in the input mfcc features
    ouput_channels : int
        Number of channels in the output of the CNN
    kernel_size : int
        Size of the kernel in the CNN
    embed_size : int
        Number of hidden units in the LSTM
    output_size : int
        Number of output classes
    training : bool
        Whether the model is in training mode or not
    
    """
    def __init__(self,input_channels, kernel_size,  output_channels,  embed_size= 192, output_size= 41, training=True) -> None:
        super(ASRMODEL, self).__init__()
        
        self.training                   = training
        
        self.augmentations              = nn.Sequential(
            PermuteBlock(),
            torchaudio.transforms.TimeMasking(200),
            torchaudio.transforms.FrequencyMasking(5),
            PermuteBlock(),
        )
        
        self.encoder                    = Encoder(input_channels, output_channels, kernel_size, padding=1, stride=1, encoder_hidden_size=embed_size)
        self.decoder                    = Decoder(embed_size*2,output_size)
        
    def forward(self,x, lengths_x):
        """This method implements a forward pass through the ASR model
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of mfcc features of shape (B, T, C)
        lengths_x : torch.Tensor
            Lengths of the input sequences of shape (B)
        Returns
        -------
        decoder_out : torch.Tensor
            Output of the decoder of shape (B, T, C)
        encoder_len : torch.Tensor
            Lengths of the output sequences of shape (B)"""
        if self.training:
            x                           = self.augmentations(x)
            
        encoder_out, encoder_len        = self.encoder(x, lengths_x)
        decoder_out                     = self.decoder(encoder_out)
        
        return decoder_out, encoder_len