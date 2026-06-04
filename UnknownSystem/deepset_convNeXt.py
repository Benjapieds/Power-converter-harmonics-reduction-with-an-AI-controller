from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor




class ModulatedConvNeXtBlock(nn.Module):
    def __init__(self, in_channels:int,  expansion:int=4):
        super().__init__()
        mid = in_channels * expansion
        self.net1 = nn.Conv1d(in_channels, in_channels, 7, padding=3, padding_mode='circular', groups=in_channels)  # depthwise 7x7
        self.net2 = nn.GroupNorm(1, in_channels)  # LayerNorm for conv (groups=1)
        self.net3 = nn.Conv1d(in_channels, mid, 1, bias=False) # Bias given by modulation
        self.net4 = nn.GroupNorm(1, mid, affine=True)  # Non-Learnable layer norm for enhacing bias effect
        self.net5 = nn.GELU()
        self.net6 = nn.Conv1d(mid, in_channels, 1)
        

    def forward(self, x: torch.Tensor, modulation_bias: torch.Tensor) -> torch.Tensor:
        #x shape (B, Cin, H)
        #modulation_biases shape (B, Cmid) for modulated layer
        skip = x 
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)

        bias = modulation_bias.unsqueeze(-1) #shape (B, Cmid, 1)
        x = x + bias #broadcasting (B, Cmid, H)

        x = self.net5(x)
        x = self.net6(x)
        x = x + skip
        return x
    
    def get_modulationList(self,
                            in_channels:int, # if x (B, C, H), should be H
                            expansion:int = 4,
                            ):
        
        mid = in_channels * expansion #expansion = 4
        return (in_channels, mid)
    
class ModulatedConvNeXtBlockV2(nn.Module):
    """
    With Adapted out_channels
    
    """
    def __init__(self, in_channels:int, out_channels:int,expansion:int=4):
        super().__init__()
        mid = in_channels * expansion
        self.net1 = nn.Conv1d(in_channels, in_channels, 7, padding=3, padding_mode='circular', groups=in_channels)  # depthwise 7x7
        self.net2 = nn.GroupNorm(1, in_channels)  # LayerNorm for conv (groups=1)
        self.net3 = nn.Conv1d(in_channels, mid, 1, bias=False) # Bias given by modulation
        self.net4 = nn.GroupNorm(1, mid, affine=True)  # Non-Learnable layer norm for enhacing bias effect
        self.net5 = nn.GELU()
        self.net6 = nn.Conv1d(mid, out_channels, 1)

    def forward(self, x: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        # x shape (B,Cin,H)
        # biases shape (B, mid)
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)

        bias = biases.unsqueeze(-1) #shape (B, mid, 1)
        x = x + bias #broadcasting (B, mid, H)

        x = self.net5(x)
        x = self.net6(x)
        
        return x
    
    def get_modulationList(self,
                            in_channels:int, # if x (B, C, H), should be H
                            expansion:int = 4,
                            ):
        
        mid = in_channels * expansion #expansion = 4
        return (in_channels, mid)
    
class ModulatedDown(nn.Module):
    def __init__(self, in_ch:int):
        super().__init__()
        self.block1 = ModulatedConvNeXtBlock(in_ch)
        self.block2 = ModulatedConvNeXtBlockV2(in_ch, in_ch * 2)
        self.norm = nn.GroupNorm(1, in_ch * 2)
        self.downsample = nn.Conv1d(in_ch * 2, in_ch * 2, 2, stride=2)

    def forward(self, x:torch.Tensor, bias1:torch.Tensor, bias2:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape (B, Cin, H)
        # bias1 shape (B, mid1) for block1
        # bias2 shape (B, mid2) for block2
        x = self.block1(x, bias1)
        x = self.block2(x, bias2)
        skip = self.norm(x)
        x = self.downsample(skip)

        return skip, x
    
    def get_modulationList(self,
                        in_ch:int):
        
        return (self.block1.get_modulationList(in_ch), self.block2.get_modulationList(in_ch))

class ModulatedUp(nn.Module):
    def __init__(self, in_ch:int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = ModulatedConvNeXtBlockV2(in_ch, in_ch // 2)  # concat with skip, so in_ch //2 is doubled
        self.norm = nn.GroupNorm(1, in_ch // 2)
        

    def forward(self, x:torch.Tensor, skip:torch.Tensor, bias:torch.Tensor) -> torch.Tensor:
        # x shape (B, Cin, H)
        # skip shape (B, Cin//2, H*2)
        # bias shape (B, mid) for block
        x = self.up(x) # shape (B, Cin//2, H*2)
        x = torch.cat([x, skip], dim=1) # shape (B, Cin, H*2)
        x = self.block(x, bias)
        x = self.norm(x)
        return x
    
    def get_modulationList(self,
                        in_ch:int):
        
        return self.block.get_modulationList(in_ch)
    

class ModulatedConvNeXtUnet(nn.Module):
    def __init__(self, input_channels:int = 1, channels:int=32, patch_len:int=15, patch_stride:int=1, patch_padding:int=7, depth:int=3):
        super().__init__()
        self.input_channels = input_channels
        self.channels = channels
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.depth = depth

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, patch_len, stride=patch_stride, padding=patch_padding, padding_mode='circular'),  # patchify stem (simplified for 32x32)
            nn.GroupNorm(1, channels))
        #[TODO] explore patching with recovering !!

        self.blks_down = nn.Sequential()
        self.blks_up = nn.Sequential()

        in_ch = channels

        for d in range(depth):
            self.blks_down.add_module("down"+str(d + 1), ModulatedDown(in_ch))

            in_ch = in_ch * 2
            

        self.bottleneck = ModulatedConvNeXtBlockV2(in_ch, in_ch * 2)
        in_ch = in_ch * 2

        for d in range(depth):
            self.blks_up.add_module("up"+str(depth-d), ModulatedUp(in_ch))
            in_ch //= 2

        self.head = nn.Conv1d(channels * 2, 1, 1)#1 as kernel size to recover image

    def get_modulationList(self, in_channels:int, channels:int):
        mod_list = []
        current_shape =channels

        for down in self.blks_down:
            mod_list.append(down.get_modulationList(current_shape))
            
            current_shape = current_shape * 2

        mod_list.append(self.bottleneck.get_modulationList(current_shape))
        current_shape = current_shape * 2

        for up in self.blks_up:
            
            mod_list.append(up.get_modulationList(current_shape))
            current_shape = current_shape // 2


        return mod_list



    def forward(self, x: torch.Tensor, modulation_list: List[torch.Tensor]) -> torch.Tensor:
        x = x.permute(0, 2, 1).contiguous() #x=(B,H,1)-> x=(B,1,H)
        x = self.stem(x)  # Apply stem to expand channels x = (B, channels, H)

        skips = []
        j = 0 #index for modulation list
        for i, down_block in enumerate(self.blks_down):
            skip, x = down_block(x, modulation_list[j], modulation_list[j+1])
            skips.append(skip)
            j += 2

       
        x = self.bottleneck(x, modulation_list[j])  # x=(B, 256, H/8) -> x=(B, 512, H/8)
        j += 1
        
        for i, up_block in enumerate(self.blks_up):

            skip = skips[-(i+1)]
            x = up_block(x, skip, modulation_list[j])
            j += 1

        # ex for depth = 3
        # After stem: torch.Size([2, 32, 600])
        # After down block 1: torch.Size([2, 64, 300])
        # After down block 2: torch.Size([2, 128, 150])
        # After down block 3: torch.Size([2, 256, 75])
        # After bottleneck: torch.Size([2, 512, 75])
        # After up block 1: torch.Size([2, 256, 150])
        # After up block 2: torch.Size([2, 128, 300])
        # After up block 3: torch.Size([2, 64, 600])
        # After up blocks final: torch.Size([2, 64, 600])
          
        h = F.tanh(self.head(x)) # x=(B, 64, H) -> h=(B, 1, H)
        return h.permute(0, 2, 1).contiguous()# h=(B, 1, H) -> h=(B, H, 1)


    def predict_step(self, batch, num_steps, pastwindow,
                     save_attention_weights=False):
        """Defined in :numref:`sec_seq2seq_training`"""
        #return the predicted sequence and attention weights stored
        #Batch is [X]
        #X shape (batch_size, pastwindow +400, 1)
       
        #Y_target shape (batch_size, 400, 1)

        #0 shot prediction
        src = batch[0]
        out = self(src)# shape (batch_size, pastwindow + 400, 1)

        return out, None #No attention weights in UNet [TODO] chage this by convolution masked shown
    
    def predict_loss(self, ref, target, pastwindow):
        """
        ref shape (batch_size, 400, 1)
        target shape (batch_size, 400, 1)

        """
        

        pred = self(ref) #shape (batch_size, pastwindow + 400, 1)
        loss = self.loss(pred, target, pastwindow) #shape (batch_size,)

        return loss
    
    def get_Jacobian(self):
        """Return a function that computes the Jacobian of predict_loss.

        For the bound method self.predict_loss, argnums=0 refers to ref,
        argnums=1 refers to target, and argnums=2 refers to pastwindow.
        """

        return torch.func.jacrev(self.predict_loss, argnums=0, has_aux=False, chunk_size=None)


############ Deepset ############################
class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, expansion=4):
        super().__init__()
        mid = channels * expansion
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3, padding_mode='circular', groups=channels),  # depthwise 7x7
            nn.GroupNorm(1, channels),  # LayerNorm for conv (groups=1)
            nn.Conv1d(channels, mid, 1),
            nn.GELU(),
            nn.Conv1d(mid, channels, 1),
        )

    def forward(self, x):
        x = x + self.net(x)
        return x 

class ConvNeXt(nn.Module):
    def __init__(self, input_channels:int = 2, channels:int=32, patch_len:int=15, patch_stride:int=7, patch_padding:int=7, depth:int=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, patch_len, stride=patch_stride, padding=patch_padding, padding_mode='circular'),  # patchify stem (simplified for 32x32)
            nn.GroupNorm(1, channels))
        #[TODO] explore patching with recovering !!


        downsampling_layers = []
        current_channels = channels

        for i in range(depth):
            stage = nn.Sequential(
                ConvNeXtBlock(current_channels),
                ConvNeXtBlock(current_channels))
            downsampling_layers.append(stage)
            down = nn.Sequential(
                nn.GroupNorm(1, current_channels),
                nn.Conv1d(current_channels, current_channels * 2, 2, stride=2))
            downsampling_layers.append(down)
            current_channels *= 2
        

        self.stage_downsampling = nn.Sequential(*downsampling_layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), #shape (B, C, 1) or [TODO] pool over the feature space instead ?
            nn.Flatten(), #shape (B, C)
            nn.GroupNorm(1, current_channels),  # final norm before aggregator
            )

        self.out_channels = current_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.stage_downsampling(x)
        return self.head(x)



class DeepSet(nn.Module):
    def __init__(self, 
                node_dim:tuple[int], # (in_c = 2, T= 400, h = 1)
                channels:int, # splitting channels of encoder start
                patch_len:int,# patching of encoder start
                patch_stride:int,
                patch_padding:int,
                depth:int, # Encoder depth
                aggregation:str, # max or mean
                hiden_dim:int, # hidden dimension of the MLP
                out_dim:int, # output head MLP dim
    ):
        super().__init__()

        self.phi = ConvNeXt(input_channels=2,
                            channels=channels,
                            patch_len = patch_len,
                            patch_stride = patch_stride,
                            patch_padding = patch_padding,
                            depth=depth,
                            ) # [TODO] adapt ConvNeXt for 1D time series, and output a vector of size L


        #[TODO] missing a bottleneck proccessor ?

        if aggregation == "mean":
            self.aggregator = lambda x: x.mean(dim=1) # (B, N, L) -> (B, L)
        elif aggregation == "max":
            self.aggregator = lambda x: x.max(dim=1).values # (B, N, L) -> (B, L)
        else:
            raise ValueError("Unknown aggregation mode")
        
        self.encoder_output_dim = self.phi.out_channels 
        print("ConvNeXt output dim:", self.encoder_output_dim)

        self.head = nn.Sequential(
            nn.Linear(self.encoder_output_dim, hiden_dim),
            nn.GELU(),
            nn.Linear(hiden_dim, hiden_dim),
            nn.GELU(),
            nn.Linear(hiden_dim, out_dim)
        )


    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, 2, T, 1)
        B, N, C, T, _ = x.shape

        x = x.squeeze(-1) # (B, N, 2, T)
        x = x.reshape(B * N, 2, T) # (B * N, 2, T)

        x = self.phi(x) # (B * N, L)
        x = x.reshape(B, N, -1) # (B, N, L)

        x = self.aggregator(x) # (B, L)

        x = self.head(x) # (B, out_dim)

        return x


######## hypernetwork #####################################

class HyperNetwork(nn.Module):
    #[TODO] normalise output cnn, normalise bias, sum and activation

    def __init__(self,
                lr:float=1e-3, #Optimizer params 
                input_channels:int=1, # Unet params
                channels:int=32, 
                patch_len:int = 15, 
                patch_stride:int = 1,
                patch_padding:int = 7,
                depth:int=3, 
                node_dim:tuple[int]=(2, 400, 1),# Deepset params
                channels_Deepset:int=32,
                patch_len_Deepset:int=16,
                patch_stride_Deepset:int = 7,
                patch_padding_Deepset:int = 7,
                depth_Deepset:int=3,
                aggregation:str="mean",
                input_dim:int=400, # Modulation needed info
                factor:float=2.0,
                ):
        super().__init__()
        self.lr = lr
        self.input_channels = input_channels
        self.channels = channels
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.depth = depth
        self.node_dim = node_dim
        self.channels_Deepset = channels_Deepset
        self.patch_len_Deepset = patch_len_Deepset
        self.patch_stride_Deepset = patch_stride_Deepset
        self.patch_padding_Deepset = patch_padding_Deepset
        self.depth_Deepset = depth_Deepset
        self.aggregation = aggregation
        self.input_dim = input_dim
        self.factor = factor
        
        #Slave Network
        self.unet = ModulatedConvNeXtUnet(
                        input_channels=input_channels,
                        channels=channels,
                        patch_len=patch_len,
                        patch_stride=patch_stride,
                        patch_padding=patch_padding,
                        depth=depth,)
        
        self.modulation_list = self.unet.get_modulationList(input_channels, channels)


        flatten_mod_in = []
        self.flatten_mod_mid = []
        for elem in self.modulation_list:
            if isinstance(elem[0], tuple):  # Down block: tuple of tuples
                for layer in elem:
                    flatten_mod_in.append(layer[0])
                    self.flatten_mod_mid.append(layer[1])
            else:  # Bottleneck or up: tuple of integers
                flatten_mod_in.append(elem[0])
                self.flatten_mod_mid.append(elem[1])

        number_bias = sum(self.flatten_mod_mid)

        print(self.modulation_list)
        print("Total number of bias to generate:", number_bias)
        print("Flatten modulation list in:", flatten_mod_in)
        print("Flatten modulation list mid:", self.flatten_mod_mid)
        #biases_list = torch.split(biases, self.flatten_mod_mid, dim=1) #list of tensors of shape (B, num_bias) for each layer
        

        #Master Network
        self.master = DeepSet(node_dim=node_dim,
                            channels=channels_Deepset,
                            patch_len=patch_len_Deepset,
                            patch_stride = patch_stride_Deepset,
                            patch_padding = patch_padding_Deepset,
                            depth= depth_Deepset,
                            aggregation=aggregation,
                            hiden_dim=number_bias // 2, 
                            out_dim=number_bias)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.norm_vec = self.get_normalization_factor(number_bias, flatten_mod_in, self.flatten_mod_mid).to(self.device) #normalization factor for each bias to stabilise training

        

    def vector_to_parameterList(self, vector):
        
        splits = torch.split(vector, self.flatten_mod_mid, dim=1) #list of tensors of shape (B, num_bias) for each layer
        return splits
    
    def get_normalization_factor(self, number_bias:int, in_list:int, mid_list:int, kernel_size:int=1) -> torch.Tensor:
        with torch.no_grad():
            
            norm_factor = torch.zeros(number_bias)
            i=0
            for j, output_channel in enumerate(mid_list):
                k = 1 / (in_list[j] * kernel_size)
                norm_factor[i:i + output_channel] = k
                i += output_channel

            return norm_factor.pow(0.5)

        
    @staticmethod
    def signed_sigmoid(x, factor=2.0):
        """
        theta_norm = getattr(self, f"theta_norm_{dset_name}")
        def signed_sigmoid(x, factor=2.0):
            return factor * (2 * torch.sigmoid(2 * x / factor) - 1)
        theta = theta * theta_norm / theta_norm.pow(2.0).mean().pow(0.5)
        theta = theta_norm * signed_sigmoid(theta / theta_norm)
        """
        return factor * (2 * torch.sigmoid( x / factor) - 1)# should have the 2* ? [TODO] check in training
    
    
    def forward(self, x, past):
        #x : end of a trajectory of shape (B,T, 1)
        #past: past trajectory of shape (B,N-1,2,T,1)

        biases = self.master(past) #shape (B, number_bias)

        biases = biases / self.norm_vec.unsqueeze(0)  # normalize biases for stable training, shape (B, number_bias)
        biases = self.signed_sigmoid(biases, factor = self.factor) #shape (B, number_bias)
        biases = biases * self.norm_vec.unsqueeze(0)  # rescale back to original range, shape (B, number_bias)


        biases_list = self.vector_to_parameterList(biases) 
        #print(biases_list) #list of blocks, each block is a list of layers, each layer is a tensor of shape (B, out_ch)
        




        #Missing the regulation of biases to stabilise training [TODO]
        # Normalize weights per layer accounting for the size of each layer
        

        x = self.unet(x, biases_list)

        return x
    def get_biases(self, past):
        biases = self.master(past) #shape (B, number_bias)

        biases = biases / self.norm_vec.unsqueeze(0)  # normalize biases for stable training, shape (B, number_bias)
        biases = self.signed_sigmoid(biases, factor = self.factor) #shape (B, number_bias)
        biases = biases * self.norm_vec.unsqueeze(0)  # rescale back to original range, shape (B, number_bias)

        return self.vector_to_parameterList(biases)
    
    def configure_optimizers(self):
        # Adam with weight decay optimizer
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def loss(self, Y_hat, Y, pastwindow, averaged=True):
        #Return MSE over a batch
        #Y_hat shape (batch_size, past_window + 400, 1)
        #Y shape (batch_size, 400, 1)

        return F.mse_loss(Y_hat, Y, reduction='mean' if averaged else 'none')
    
    def validation_step(self, batch, pastwindow):
        #batch is [x, past, target]
        Y_hat = self(*batch[:-1])
        loss = self.loss(Y_hat, batch[-1], pastwindow)
        return loss
    
    def training_step(self, batch, pastwindow):
        #batch is [x, past, target]
        l = self.loss(self(*batch[:-1]), batch[-1], pastwindow)
        return l
    
    def predict_step(self, batch, num_steps, pastwindow,
                     save_attention_weights=False):
        """
        batch is [x, past]

        
        """
        prediction = self.forward(*batch) #shape (B, pastwindow + 400, 1)
        return prediction, None
    
    def predict_loss(self, x, target, pastwindow, past_context):
        """
        ref shape (batch_size, 400, 1)
        target shape (batch_size, 400, 1)

        """
        
        prediction = self.forward(x, past_context) #shape (batch_size, pastwindow + 400, 1)
        loss = self.loss(prediction, target, pastwindow) #shape (batch_size,)
        return loss
        
    
    def get_Jacobian(self):
        """Return a function that computes the Jacobian of predict_loss.

        For the bound method self.predict_loss, argnums=0 for x.
        """

        return torch.func.jacrev(self.predict_loss, argnums=0, has_aux=False, chunk_size=None)
    
    def THD(self, Y_hat, Y, averaged=True): #[TODO]
        raise NotImplementedError
    
    def count_parameters(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    hypernetwork = HyperNetwork(lr= 1e-3,
                            input_channels=1,
                            channels=8,
                            patch_len=15,
                            patch_stride=1,
                            patch_padding=7,
                            depth=3,
                            node_dim=(2, 448, 1),
                            channels_Deepset=32,
                            patch_len_Deepset=16,
                            patch_stride_Deepset=7,
                            patch_padding_Deepset=7,
                            depth_Deepset=3,
                            aggregation="max",
                            input_dim=448).to(device)
    
    x = torch.randn(8, 448, 1).to(device) # (B, T, 1)
    past = torch.randn(8, 5, 2, 448, 1).to(device) # (B, N-1, 2, T, 1)
    out = hypernetwork(x, past)
    print(out.shape) # (B, T, 1)
    print("Total number of parameters in HyperNetwork:", hypernetwork.count_parameters())
