"""
    File name: utils.py
    Author: Gabriel Moreira
    Date last modified: 3/28/2022
    Python Version: 3.9.10
"""

import os
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

def time_mask(src, seq_len, device="cuda"):
    """
        Creates a mask where mask[b,t] = 0 if seq_len[b] > t
        Parameters:
        ___________
        src (torch.tensor): shape=(batch_size, max_seq_len, *)
        seq_len (torch.tensor): shape=(batch_size, 1)

        Returns:
        ________
        mask (torch.tensor): shape=(batch_size, max_seq_len)
    """ 
    mask = torch.zeros(src.shape[0], src.shape[1] + 1, dtype=src.dtype, device=src.device)
    mask[(torch.arange(src.shape[0]), seq_len)] = 1
    mask = mask.cumsum(dim=1)[:, :-1]  # remove the superfluous column

    mask = mask.to(device)
    return mask > 0

    
def dynamic_batching(sequence, device="cuda"):
    """
        Parameters:
        ___________
        sequence (list): [torch.tensor items of shape (sequence_length, *)]

        Returns:
        ________
        split_seq (list): List of len: max(sequence_length) with torch.tensor
        items of shape (batch_size, *)
        seq_len (list): List of sequence lengths of each item in sequence
    """        
    # Store original sequence lengths
    seq_len = torch.tensor([batch_item.shape[0] for batch_item in sequence]).to(device)

    # Obtain a single torch.tensor of shape (largest_sequence_length, batch_size, *)
    padded_sequence = pad_sequence(sequence)

    # Split the previous tensor along dim=0 (time axis) into a list of 
    # length=largest_sequence_length containing sub-tensors each 
    # of shape (batch_size, *)
    split_seq = torch.split(padded_sequence, [1]*padded_sequence.shape[0])

    # Reduce the dimension used to split the tensor
    split_seq = list(map(lambda x : torch.squeeze(x,0).to(device), split_seq))
    
    return split_seq, seq_len


def get_batched_instructions(batch, device="cuda"):
    """
        Parameters:
        ___________
        batch (list): [dict items with task data]

        Returns:
        ________
        batched_instruction (torch.tensor): shape (batch_size, max language instruction length)
    """
    batched_instruction = []
    seq_len = []
    for b in batch:
        instruction = b['num']['lang_instr']

        gold_instruction = torch.tensor([item for sublist in instruction for item in sublist], dtype=torch.long)

        seq_len.append(len(gold_instruction))

        batched_instruction.append(gold_instruction)

    batched_instruction = pad_sequence(batched_instruction, batch_first=True).to(device)
    seq_len = torch.tensor(seq_len).to(device)

    return batched_instruction, seq_len


def getNumTrainableParams(network):
    '''
    '''
    num_trainable_parameters = 0
    for p in network.parameters():
        num_trainable_parameters += p.numel()
    return num_trainable_parameters




class Tracker:
    def __init__(self, metrics, filename, load=False):
        '''
        '''
        self.filename = os.path.join(filename, 'tracker.csv')

        if load:
            self.metrics_dict = self.load()
        else:        
            self.metrics_dict = {}
            for metric in metrics:
                self.metrics_dict[metric] = []


    def update(self, **args):
        '''
        '''
        for metric in args.keys():
            assert(metric in self.metrics_dict.keys())
            self.metrics_dict[metric].append(args[metric])

        self.save()


    def isLarger(self, metric, value):
        '''
        '''
        assert(metric in self.metrics_dict.keys())
        return sorted(self.metrics_dict[metric])[-1] < value


    def isSmaller(self, metric, value):
        '''
        '''
        assert(metric in self.metrics_dict.keys())
        return sorted(self.metrics_dict[metric])[0] > value


    def save(self):
        '''
        '''
        df = pd.DataFrame.from_dict(self.metrics_dict)
        df = df.set_index('epoch')
        df.to_csv(self.filename)


    def load(self):
        '''
        '''
        df = pd.read_csv(self.filename)  
        metrics_dict = df.to_dict(orient='list')
        return metrics_dict


    def __len__(self):
        '''
        '''
        return len(self.metrics_dict)