# ASRMODEL
This repository implements an Automatic Speech Recognition Model. The model predicts the phonemes in a recording. An audio of a person saying a word passes through the network, which outputs the phonetic transcription. The model makes use of RNNs and Connection Temporal Classification dynamic programming algotithm to generate labels.  
The model consists of the following components:
* **1-D CNN layer** 1D CNNs capture the structural dependence between adjacent vectors in the input.
* **Bidirectional LSTM layers (Bi-LSTMS)** to capture long-term contextual dependencies.

* **Pyramidal Bi-LSTMS (PBLSTMs)** reduce the time resolution of the input by a factor of 2.

### Contents
#### src
This folder contains all the training and evaluation scripts as well as the utility functions  
* **main.py** Use it to run training, validation, and inference
* **model.py** File contains the LAS model. It contains the Listener, Attention and Speller Module.
* **train_test.py** File contains the train, validation and test code.
* **dataloader.py** File contains code to load data for training the model.

#### config
This folder contains the config files that define the hyperparameters and other variables useful to run the training and evaluation scripts
* **config.yaml** File contains model and dataset configurations 
#### Notebook
* **notebook.ipynb** Notebook for running the model. The data should be placed in a data folder with a split for training, validation and testing

### Usage
All scripts should be executed from the parent folder LAS.  
Download and unzip the dataset into a data folder.  
Add a wandb key to track the model using wandb.
Use ```python main.py``` to run the model. Use the config.yaml file to change model parameters
