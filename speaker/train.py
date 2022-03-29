"""
    File name: train.py
    Author: Gabriel Moreira
    Date last modified: 3/28/2022
    Python Version: 3.9.10
"""

import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import math
from trainer import Trainer
from loader import TaskLoader
from rnn import EncoderLSTM, DecoderLSTM

from utils import getNumTrainableParams

"""
    FILES/PATHS
"""
data_path        = './data/json_feat_2.1.0'
feat_pt_filename = 'feat_conv.pt'
splits_path      = './data/splits/oct21.json'
vocab_path       = "./exp/seq2seq_im_mask/pp.vocab"

"""
    CONFIGURATION
"""
cfg = {'name'       : 's1',
       'seed'       : 1,
       'epochs'     : 100,
       'batch_size' : 1,
       'lr'         : 0.001,
       'resume'     : False}


if __name__ == '__main__':
    # If experiment folder doesn't exist create it
    if not os.path.isdir(cfg['name']):
        os.makedirs(cfg['name'])
        print("Created experiment folder : ", cfg['name'])
    else:
        print(cfg['name'], "folder already exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # LOAD ALFRED'S VOCABULARY
    vocab = torch.load(vocab_path)

    # TASK LOADERS TO BE USED DURING TRAINING
    train_loader = TaskLoader(data_path, splits_path, 'train',      cfg['batch_size'], shuffle=True)
    dev_loader   = TaskLoader(data_path, splits_path, 'valid_seen', cfg['batch_size'], shuffle=True)

    # MODEL PARAMS
    action_vocab_size     = len(vocab['action_low'])
    word_vocab_size       = len(vocab['word'])
    action_embedding_size = 256
    word_embedding_size   = 512
    visual_embedding_size = 512 # Alfred's ResNet18 features are (T,49,512)
    hidden_size           = 1024

    speaker_encoder = EncoderLSTM(action_vocab_size, action_embedding_size, visual_embedding_size, hidden_size, 0.1).to(device)
    speaker_decoder = DecoderLSTM(word_vocab_size,     word_embedding_size, hidden_size, dropout_ratio=0.1, use_input_att_feed=True).to(device)

    # OPTIMIZATION
    encoder_optimizer = optim.SGD(speaker_encoder.parameters(), lr=cfg['lr'])
    decoder_optimizer = optim.SGD(speaker_decoder.parameters(), lr=cfg['lr'])
    encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, T_max=(len(train_loader) * cfg['epochs']))
    decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=(len(train_loader) * cfg['epochs']))
    criterion         = nn.NLLLoss(ignore_index=0)

    trainer = Trainer(speaker_encoder,
                      speaker_decoder,
                      encoder_optimizer,
                      decoder_optimizer,
                      encoder_scheduler,
                      decoder_scheduler,
                      vocab,
                      cfg['epochs'],
                      criterion,
                      train_loader, 
                      dev_loader,
                      device,
                      cfg['name'],
                      cfg['resume'])

    # Verbose
    print('Experiment ' + cfg['name'])
    print('Running on', device)
    print('Train - {} batches of size {}'.format(math.ceil(len(train_loader)/cfg['batch_size']), cfg['batch_size']))
    print('  Val - {} batches of size {}'.format(math.ceil(len(dev_loader)/cfg['batch_size']), cfg['batch_size']))
    print('Number of trainable parameters (encoder): {}'.format(getNumTrainableParams(speaker_encoder)))
    print('Number of trainable parameters (decoder): {}'.format(getNumTrainableParams(speaker_decoder)))
    print(speaker_encoder)
    print(speaker_decoder)

    trainer.fit()




    


