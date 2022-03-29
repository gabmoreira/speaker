"""
    File name: loader.py
    Author: Gabriel Moreira
    Date last modified: 3/28/2022
    Python Version: 3.9.10
"""

import os
import json
import random

import torch

from tqdm import trange

    
class TaskLoader:
    def __init__(self, data_path, splits_file, split, batch_size, shuffle=False):
        """
        """
        self.data_path  = data_path
        self.split      = split
        self.shuffle    = shuffle
        self.batch_size = batch_size

        self.feat_pt_filename = 'feat_conv.pt'

        # Open JSON and load with dictionary with train/dev/test splits
        with open(splits_file) as f :
            self.splits = json.load(f)
    
        # List of tasks (not task dicts !) corresponding to specified split
        self.tasks = self.splits[split]

        self.num_tasks = len(self.tasks)
        
    
    def __len__(self):
        '''
            Number of tasks in the split
        '''
        return self.num_tasks
    
        
    def __getitem__(self, i):
        '''
            Get the i-th task dictionary in the split
        '''
        return self.load_task_json(self.tasks[i])


    def iterate(self):
        '''
            Iterates over batches, not elements.
            Each batch is a list of task dictionaries
            loaded from JSON files.
        '''
        if self.shuffle:
            random.shuffle(self.tasks)

        for i in range(0, len(self.tasks), self.batch_size):
            batch_tasks = self.tasks[i:i+self.batch_size]

            batch_task_dicts = []
            for task in batch_tasks:
                task_dict = self.load_task_json(task)
                if task_dict is not None:
                    batch_task_dicts.append(task_dict)

            # Loads ResNet features torch.tensor
            batch_img_features = self.featurize_batch(batch_task_dicts)

            yield i, batch_task_dicts, batch_img_features


    def featurize_batch(self, batch):
        # Batch is a list of task dicts
        features = []
        for b in batch:
            # b is a task dict
            # ResNet18 features are torch.tensor of shape (T,512,7,7)
            # get_img_features() flattens it to (T,512,49)
            img_features = self.get_img_features(b)
            # Permute to have (T,49,512)
            img_features = torch.permute(img_features, [0,2,1])
            
            num_low_actions = len(b['plan']['low_actions']) + 1  # +1 for additional stop action
            num_feat_frames = img_features.shape[0]

            # One image/feature tensor per low action
            # Original dataset has more images and features
            if (num_low_actions == num_feat_frames):
                features.append(img_features)

        return features


    def get_img_features(self, task_dict):
        task_root_path = os.path.join(self.data_path, task_dict['split'], *(task_dict['root'].split('/')[-2:]))

        img_features = torch.load(os.path.join(task_root_path, self.feat_pt_filename))
        img_features = torch.flatten(img_features, start_dim=2, end_dim=-1)

        return img_features
        

    def load_task_json(self, task):
        '''
            Load preprocessed JSON from disk (adapted from ALFRED)
        '''
        json_path = os.path.join(self.data_path, task['task'], '%s' % 'pp', 'ann_%d.json' % task['repeat_idx'])

        task_dict = None
        try:
            with open(json_path) as f:
                task_dict = json.load(f)
        except:
            print('Problem loading {}'.format(json_path))

        return task_dict
