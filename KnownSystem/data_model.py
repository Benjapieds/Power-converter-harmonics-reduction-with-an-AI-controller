import torch
import random

from base_model import HyperParameters


class PlantDataBis(HyperParameters):
    def __init__(self, batch_size, pastwindow = 48, root = 'C:/Users/keutg/OneDrive - Universite de Liege/M2/TFE/Codes/dataset_onesystem', num_workers = 4, seed = 42):
        self.save_hyperparameters()
        self.tensor_train = torch.load(root + "/train.pt") # shape (637000, 2, 400)
        self.tensor_val = torch.load(root + "/val.pt") # shape (79625, 2, 400)
        self.tensor_test = torch.load(root + "/test.pt") # shape (79625, 2, 400)

        self.normalization = 0.9740
        self.data_augmented = False

    def get_dataloader(self, type):
        if type == 'train':
            tensors = self.tensor_train
        elif type == 'val':
            tensors = self.tensor_val
        elif type == 'test':
            tensors = self.tensor_test
        else:
            raise ValueError(f"Unknown dataloader type: {type}")

        return self.get_tensorloader(tensors, type)

    def train_dataloader(self):
        return self.get_dataloader(type='train')
    
    def val_dataloader(self):
        return self.get_dataloader(type='val')
    
    def test_dataloader(self):
        return self.get_dataloader(type='test')
    
    def get_tensorloader(self, tensors, type):
        
        X = tensors[:, 0, :].float() / self.normalization
        Y = tensors[:, 1, :].float() / self.normalization

        X = X.unsqueeze(-1) # shape (N, 400, 1)
        Y = Y.unsqueeze(-1) # shape (N, 400, 1)

        dataset = torch.utils.data.TensorDataset(X, Y)

        shuffle = type == 'train'
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle = shuffle,
                                           generator = generator,
                                           num_workers = self.num_workers)




class PlantData(HyperParameters):
    def _transform(self, arrays):
        """
        arrays[0]: X shape (156000, 400, 1)
        arrays[1]: Y_targeted shape (156000, 400, 1)

        out: arrays[1]: Y_targeted shape (156000, 400, 1) with interpolation
        arrays[0]: X shape (156000, pastwindow + 400, 1) unchanged
        """
        
        #Keep last period
        Y_targeted = arrays[1][:, 200:] #shape (156000, 200, 1)
        X = arrays[0]
        X = torch.cat([X[:, -self.pastwindow:], X], dim=1) #shape (156000, pastwindow + 400, 1)

        # Interpolate (periodic assymption)
        shifted = torch.roll(Y_targeted, shifts=-1, dims=1)
        average = (Y_targeted + shifted) / 2.0
        Y_targeted = torch.stack((Y_targeted, average), dim=2).reshape(
            Y_targeted.shape[0], -1, Y_targeted.shape[2])  # shape (156000, 400, 1)

        #Normalization startegy [TODO]
        max_normalisation = 0.9740
        Y_targeted = Y_targeted / max_normalisation
        X = X / max_normalisation
        
        return X, Y_targeted
    
    def _transform_without_past(self, arrays):
        """
        arrays[0]: X shape (156000, 400, 1)
        arrays[1]: Y_targeted shape (156000, 400, 1)

        out: arrays[1]: Y_targeted shape (156000, 400, 1) with interpolation
        arrays[0]: X shape (156000, pastwindow + 400, 1) unchanged
        """
        
        #Keep last period
        Y_targeted = arrays[1][:, 200:] #shape (156000, 200, 1)
        X = arrays[0]
        #X = torch.cat([X[:, -self.pastwindow:], X], dim=1) #shape (156000, pastwindow + 400, 1)

        # Interpolate (periodic assymption)
        shifted = torch.roll(Y_targeted, shifts=-1, dims=1)
        average = (Y_targeted + shifted) / 2.0
        Y_targeted = torch.stack((Y_targeted, average), dim=2).reshape(
            Y_targeted.shape[0], -1, Y_targeted.shape[2])  # shape (156000, 400, 1)

        #Normalization startegy [TODO]
        max_normalisation = 0.9740
        Y_targeted = Y_targeted / max_normalisation
        X = X / max_normalisation
        
        return X, Y_targeted

    def _download(self, root, format = 'pt'):
        # list of (list of ( X[400], 1per; Y[400], 2per ))
        try:
            return torch.load(root, map_location='cpu', weights_only=True) # new torch version
        except TypeError:
            
            return torch.load(root, map_location='cpu') 
    
    def _create_arrays(self, data, format = 'list', seed = 42):
        #List of 156000 elements
        #Each one is a list of 2 elements
        #each one is a tensor of shape 400
        if format == 'list':
            # Shuffle samples
            random.Random(seed).shuffle(data)

            #goal: create 2 tensors of shape (156000, 400, 1) for X and Y
            X_list = []
            Y_target_list = []

            for step in data:
                X_list.append(step[0].unsqueeze(-1)) #shape (400, 1)
                Y_target_list.append(step[1].unsqueeze(-1))
            
            X = torch.stack(X_list).float() #shape (156000, 400, 1)
            Y_target = torch.stack(Y_target_list).float() #shape (156000, 400, 1)

            return X, Y_target
        
        elif format == 'full':
            if torch.is_tensor(data) and data.ndim >= 3 and data.shape[1] == 2:
                X = data[:, 0]
                Y_target = data[:, 1]
            else:
                X, Y_target = data #shape(156000, 400)
            
            # Shuffle samples while keeping X and Y paired using the same seed
            n_samples = X.shape[0]
            indices = list(range(n_samples))
            random.Random(seed).shuffle(indices)
            
            X = X[indices]
            Y_target = Y_target[indices]
            
            # Unsqueeze to add last dimension
            X = X.unsqueeze(-1).float()  # shape (156000, 400, 1)
            Y_target = Y_target.unsqueeze(-1).float()  # shape (156000, 400, 1)
            
            return X, Y_target
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
    def __init__(self, 
                batch_size,
                pastwindow = 400,
                num_train = 13000,
                num_val = 1625,
                num_test = 1625,
                root = './dataset_bis/output.pt',
                expand_root = None,
                format = 'list',
                num_workers = 4,
                seed = 42):
        
        self.save_hyperparameters()
        self.data_augmented = True if expand_root is not None else False

       
        self.arrays = self._transform_without_past(self._create_arrays(self._download(root, format), format, seed))
        

        if expand_root is not None:
            self.expanded = self._transform_without_past(self._create_arrays(self._download(expand_root, format), format, seed))
        else:
            self.expanded = None
        #download
        #numpy array => torch tensor
        #transform apply on torch tensor
        #create the memory object of data
        #need arrays as torch tensors

        #need to define self.arrays
        #load the data
        #transform them
        #Bien shuffle la representaton avant self.arrays

    def get_dataloader(self, type):
        """Defined in :numref:`subsec_loading-seq-fixed-len`"""
        if type == 'train':
            idx = slice(0, self.num_train)
        elif type == 'val':
            idx = slice(self.num_train, self.num_train + self.num_val)
        elif type == 'test':
            idx = slice(self.num_train + self.num_val, None)
        else:
            raise ValueError(f"Unknown dataloader type: {type}")

        return self.get_tensorloader(self.arrays, type, idx, self.expanded)    

    def train_dataloader(self):
        return self.get_dataloader(type='train')
    
    def val_dataloader(self):
        return self.get_dataloader(type='val')
    
    def test_dataloader(self):
        return self.get_dataloader(type='test')
    
    def get_tensorloader(self, tensors, type, indices=slice(0, None), expand_tensor = None):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        
        tensors = tuple(a[indices] for a in tensors)

        #here, implement the 2X and 1X ratio
        if expand_tensor is not None:
            expand_tensor = tuple(a[indices] for a in expand_tensor)
            if type == 'train':
                tensors = tuple(torch.cat([t, t, e], dim=0) for t, e in zip(tensors, expand_tensor)) #2times more smooth data than expanded
            else:
                tensors = tuple(torch.cat([t, e], dim=0) for t, e in zip(tensors, expand_tensor)) #1time more smooth data than expanded
            
        
        dataset = torch.utils.data.TensorDataset(*tensors)

        shuffle = True if type == 'train' else False
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle = shuffle,
                                           generator = generator,
                                           num_workers = self.num_workers)
    
    def get_bounds(self):
        """
        Return the bounds of the input space.

        return: tuple of (lower_bound, upper_bound) where each is a numpy array of shape (400,)
        
        """
        #[TODO] change here if past window in dataloader
        upper_bounds = torch.max(self.arrays[0], dim=0).values.squeeze(-1) #shape (400,)
        lower_bounds = torch.min(self.arrays[0], dim=0).values.squeeze(-1) #shape (400,)

        Y_target_upper_bounds = torch.max(self.arrays[1], dim=0).values.squeeze(-1) #shape (400,)
        Y_target_lower_bounds = torch.min(self.arrays[1], dim=0).values.squeeze(-1) #shape (400,)
        return lower_bounds.numpy(), upper_bounds.numpy(), Y_target_lower_bounds.numpy(), Y_target_upper_bounds.numpy()



if __name__ == "__main__":
    root = "COMPLETE HERE"
    batchsize = 32
    pastwindow=48

    pdb = PlantDataBis(batch_size=batchsize, pastwindow=pastwindow, root=root)

    train_dataloader = pdb.train_dataloader()
    val_dataloader = pdb.val_dataloader()
    test_dataloader = pdb.test_dataloader()

    print(len(train_dataloader))
    print(len(val_dataloader))
    print(len(test_dataloader))

    train_bach = next(iter(train_dataloader))
    print(type(train_bach))

    xbatch = train_bach[0]
    ybatch = train_bach[1]

    print(xbatch.shape)
    print(ybatch.shape)

    import matplotlib.pyplot as plt

    for i in range(4):

        plt.plot(xbatch[i].squeeze().numpy(), label=f'X[{i}]')
        plt.plot(ybatch[i].squeeze().numpy(), label=f'Y[{i}]')
    plt.legend()
    plt.show()

    

    
    
       



    


