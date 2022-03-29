'''
    File name: trainer.py
    Author: Gabriel Moreira
    Date last modified: 03/08/2022
    Python Version: 3.7.10
'''

import os
import torch
from tqdm import tqdm
from eval import evaluate
from utils import *
import math


class Trainer:
    def __init__(
        self,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        encoder_scheduler,
        decoder_scheduler,
        vocab,
        epochs,
        criterion,
        train_loader,
        dev_loader,
        device,
        name,
        resume):

        self.encoder           = encoder
        self.decoder           = decoder
        self.vocab             = vocab
        self.epochs            = epochs
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.encoder_scheduler = encoder_scheduler
        self.decoder_scheduler = decoder_scheduler
        self.criterion         = criterion
        self.train_loader      = train_loader
        self.dev_loader        = dev_loader
        self.device            = device
        self.name              = name
        self.start_epoch       = 1

        self.tracker = Tracker(['epoch',
                                'train_loss'], name, load=resume)

        if resume:
            self.resume_checkpoint()


    def fit(self):
        '''
            Fit model to training set over #epochs
        '''
        is_best  = False
        # Iterate over number of epochs
        for epoch in range(self.start_epoch, self.epochs+1):
            train_loss = self.train_epoch()

            self.validate_epoch()

            self.epoch_verbose(epoch, train_loss)

            # Check if better than previous models
            if epoch > 1:
                is_best = self.tracker.isSmaller('train_loss', train_loss)

            self.tracker.update(epoch=epoch,
                                train_loss=train_loss)

            self.save_checkpoint(epoch, is_best)


    def train_epoch(self):
        '''
            Train model for ONE epoch
        '''
        self.encoder.train()
        self.decoder.train()

        # Progress bar over the current epoch
        batch_bar = tqdm(total=math.ceil(len(self.train_loader)/self.train_loader.batch_size), dynamic_ncols=True, desc='Train') 

        # Cumulative loss over all batches or Avg Loss * num_batches
        total_loss_epoch = 0
        i = 0
        # Iterate one batch at a time
        for i_batch, batch, img_features in self.train_loader.iterate():
            if i_batch % 512 == 0:
                self.validate_epoch()

            self.encoder.train()
            self.decoder.train()
            batch_size = len(batch)

            actions = [torch.tensor(self.vocab['action_low'].word2index([a['discrete_action']['action'] \
            for a in batch[i]['plan']['low_actions']]) + [2]).to(self.device) for i in range(batch_size)]

            # Get instructions as a torch.tensor shape=(batch_size, max instruction length)
            # instructions indexed to vocab['word']
            gold_instructions, instr_seq_len = get_batched_instructions(batch, device="cuda")
            # img_features will be a list [ torch.tensor's of shape(batch_size, 7*7, 512) ]
            img_features, img_seq_len = dynamic_batching(img_features, device="cuda")  
            # actions will be a list [ torch.tensor's of shape(batch_size, action_embedding_size) ]
            actions, actions_seq_len  = dynamic_batching(actions, device="cuda") 

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            # Encode all time steps of the input
            ctx, decoder_init, c = self.encoder(actions, img_features, img_seq_len)
            
            # Mask context according to sequence length
            # of each element of the mini-batch
            mask           = time_mask(ctx, actions_seq_len)
            decoder_input  = torch.full((batch_size, 1), self.vocab['word'].word2index("<<seg>>")).to(self.device)
            decoder_hidden = decoder_init
            loss_batch     = 0.0

            for t in range(gold_instructions.shape[1]):
                # decoder_hidden - Hidden (batch_size, hidden_dim)
                # c_1   - CEC (batch_size, hidden_dim)
                # alpha - Attention (batch_size, hidden_dim)
                # logit - (batch_size, word_vocab_size)
                decoder_hidden, c, alpha, logit = self.decoder(decoder_input, decoder_hidden, c, ctx, ctx_mask=mask)
                decoder_input = gold_instructions[:,t]
                loss_batch += self.criterion(logit, gold_instructions[:,t]) 

            loss_batch.backward()
            self.decoder_optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_scheduler.step()
            self.encoder_scheduler.step()

            # Performance metrics
            total_loss_epoch += loss_batch / gold_instructions.shape[1]
            avg_loss_epoch    = float(total_loss_epoch / (i + 1))

            # Performance tracking verbose
            batch_bar.set_postfix(
                avg_train_loss="{:.04f}".format(avg_loss_epoch),
                enc_lr="{:.04f}".format(float(self.encoder_optimizer.param_groups[0]['lr'])))
            batch_bar.update()
            i += 1
        batch_bar.close()

        return avg_loss_epoch


    def validate_epoch(self):
        '''
            Validation
        '''
        # Set model to evaluation mode
        self.encoder.eval()
        self.decoder.eval()

        # Do not store gradients 
        with torch.no_grad():
            # Get batches from DEV loader
            for i_batch, batch, img_features in self.dev_loader.iterate():
                if i_batch > self.dev_loader.batch_size*3:
                    break
                x, y = evaluate(self.encoder, self.decoder, self.vocab, [batch[0]], [img_features[0]])
                print("[Decoded]: {}\n---\n[Truth]: {}\n--------------------".format(" ".join(x)," ".join(y[0])))


    def save_checkpoint(self, epoch, is_best):
        '''
            Save model dict and hyperparams
        '''
        state = {"epoch": epoch,
                 "encoder": self.encoder,
                 "decoder": self.decoder,
                 "encoder_optimizer": self.encoder_optimizer,
                 "decoder_optimizer": self.decoder_optimizer,
                 "encoder_scheduler": self.encoder_scheduler,
                 "decoder_scheduler": self.decoder_scheduler }

        # Save checkpoint to resume training later
        checkpoint_path = os.path.join(self.name, "checkpoint.pth")
        torch.save(state, checkpoint_path)
        print('Checkpoint saved: {}'.format(checkpoint_path))

        # Save best model weights
        if is_best:
            e_best_path = os.path.join(self.name, "e_best_weights.pth")
            d_best_path = os.path.join(self.name, "d_best_weights.pth")
            torch.save(self.encoder.state_dict(), e_best_path)
            torch.save(self.decoder.state_dict(), d_best_path)
            print("Saving : {},{}".format(e_best_path,d_best_path))


    def resume_checkpoint(self):
        '''
        '''
        resume_path = os.path.join(self.name, "checkpoint.pth")
        print("Loading checkpoint: {} ...".format(resume_path))

        checkpoint             = torch.load(resume_path)
        self.start_epoch       = checkpoint["epoch"] + 1
        self.encoder           = checkpoint["encoder"]
        self.decoder           = checkpoint["decoder"]
        self.encoder_optimizer = checkpoint["encoder_optimizer"]
        self.decoder_optimizer = checkpoint["decoder_optimizer"]
        self.encoder_scheduler = checkpoint["encoder_scheduler"]
        self.decoder_scheduler = checkpoint["decoder_scheduler"]

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    
    def epoch_verbose(self, epoch, train_loss):
        log = "\nEpoch: {}/{} summary:".format(epoch, self.epochs)
        log += "\n       Avg train NLL loss  |  {:.6f}".format(train_loss)
        print(log)