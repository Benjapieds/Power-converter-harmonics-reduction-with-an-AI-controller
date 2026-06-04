
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

from base_model import HyperParameters

class FullData(HyperParameters):

    def _construct_system(self, system_folder:str, trajectory_dim:int, is_train:bool):
        merged = None
        for path in system_folder:
            tensor = torch.load(Path(path) / "output_all.pt", map_location='cpu', weights_only=True) # shape (system_tensors, 2, 400)
            #truncated
            number_trajectories = tensor.shape[0] // trajectory_dim
            tensor = tensor[:number_trajectories * trajectory_dim]
            #Keep normalisation value across systems
            self.max_value = max(self.max_value, torch.abs(tensor).max().item())
            #shuffle system
            indices = torch.randperm(tensor.shape[0])
            tensor = tensor[indices]
            #Transform into (B, N, 2, tensor_shape)
            tensor = torch.stack(torch.split(tensor, trajectory_dim, dim=0), dim=0).contiguous()
            if is_train == False:
                #take only 20%
                tensor = tensor[:int(0.2 * tensor.shape[0])]
            if merged is None:
                merged = tensor
            else:
                merged = torch.cat((merged, tensor), dim=0) # shape (num_trajectories, trajectory_dim, 2, tensor_shape)
        #shuffle across systems
        indices = torch.randperm(merged.shape[0])
        merged = merged[indices]
        return merged

    def __init__(self, 
                batch_size:int,
                root_dirs:List[str],
                trajectory_dim:int,
                past_window:int,
                num_workers:int = 4,
                seed:int=42,
                num_train:int = 10,
                num_val:int = 2,
                ):
        
        self.save_hyperparameters()
        torch.manual_seed(seed)
        self.max_value = 0.
        Systems_n = 1 # TODO: change to 7
        
        chunks = [root_dirs[i:i+Systems_n] for i in range(0, len(root_dirs), Systems_n)] # shape (num_chunks, Systems_n)
        if len(chunks) != (num_train + num_val):
            raise ValueError(f"Number of chunks ({len(chunks)}) does not match num_train + num_val ({num_train + num_val}). Please check the number of root_dirs (7 str in a chunk)")
        #shuffle systems
        random.seed(seed)
        random.shuffle(chunks)

        #split
        train_chunks = chunks[:num_train] # shape (num_train, Systems_n)
        val_chunks = chunks[num_train:] # shape (num_val, Systems_n)
        train_dirs = [root_dir for chunk in train_chunks for root_dir in chunk] # shape (num_train * Systems_n)
        val_dirs = [root_dir for chunk in val_chunks for root_dir in chunk] # shape (num_val * Systems_n)

        self.datas_val = self._construct_system(val_dirs, trajectory_dim, False)
        self.datas_train = self._construct_system(train_dirs, trajectory_dim, True)
        self.max_value = torch.tensor(self.max_value)

    def get_dataloader(self, type):
        if type == 'train':
            dat = torch.utils.data.DataLoader(self.datas_train,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              num_workers=self.num_workers,)
        elif type == 'val':
            dat = torch.utils.data.DataLoader(self.datas_val,
                                              batch_size=self.batch_size,
                                              shuffle=False,
                                              num_workers=self.num_workers,)
        else:
            raise ValueError(f"Unknown dataloader type: {type}")

        return dat
    
    def train_dataloader(self):
        return self.get_dataloader(type='train')
    
    def val_dataloader(self):
        return self.get_dataloader(type='val')
    
    def get_bounds(self, metadata_path):
        raise NotImplementedError
    
if __name__ == "__main__":

    path = [r"COMPLETE\dataset0",
            r"COMPLETE\dataset1",
            r"COMPLETE\dataset2",
            r"COMPLETE\dataset3",
            r"COMPLETE\dataset4",
    ]
    
    from Plant_torch import Interpolate
   
    for p in path:
        half_input_sequence_bis = []
        half_sample_output = []

        for i in range(81):
            chunk_input, chunk_output = torch.load(Path(p) / f"output_{i}.pt", map_location='cpu', weights_only=True)
            half_input_sequence_bis.append(chunk_input)
            troncated = chunk_output[:, 200:]

            
            chunck_output = Interpolate(troncated)
            half_sample_output.append(chunck_output)
        merged_half_input_sequence_bis = torch.cat(half_input_sequence_bis, dim=0)
        merged_half_sample_output = torch.cat(half_sample_output, dim=0)
        torch.save([merged_half_input_sequence_bis, merged_half_sample_output], Path(p) / "output_all.pt")
            

    for p in path:
        l = torch.load(Path(p) / "output_all.pt", map_location='cpu', weights_only=True)
        X = l[0] # shape (B, 400)
        Y = l[1] # shape (B, 400)
        t = torch.stack((X, Y), dim=1) # shape (B, 2, 400)
        print(f"Loaded tensor from {p} with shape {t.shape}")
        torch.save(t, Path(p) / "output_all.pt")

    fd = FullData(
        batch_size=512,
        root_dirs= path,
        trajectory_dim=5,
        past_window=48,
        num_workers=1,
        num_train=4,
        num_val=1,
    )
    train_loader = fd.train_dataloader()
    val_loader = fd.val_dataloader()

    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")  # should be (256, 5, 2, 400)
        first_trajectory = batch[0, :,:,:]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        scale = fd.max_value.item()
        print(scale)

        axes[0, 0].plot(first_trajectory[0, 0].numpy(), label='x')
        axes[0, 0].legend()

        axes[0, 1].plot(first_trajectory[0, 1].numpy(), label='y')
        axes[0, 1].legend()

        axes[1, 0].plot((first_trajectory[0, 0] / scale).numpy(), label='x / max_value')
        axes[1, 0].legend()

        axes[1, 1].plot((first_trajectory[0, 1] / scale).numpy(), label='y / max_value')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

            

    




    
    
       



    


