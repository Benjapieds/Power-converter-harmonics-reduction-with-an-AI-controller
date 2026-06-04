import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import Seq2Seq

class ConvNeXtBlock(nn.Module):
    """
    With skip connection
    
    """
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
    
class ConvNeXtBlockV2(nn.Module):
    """
    With Adapted out_channels
    
    """
    def __init__(self, in_channels, out_channels,expansion=4):
        super().__init__()
        mid = in_channels * expansion
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 7, padding=3, padding_mode='circular', groups=in_channels),  # depthwise 7x7
            nn.GroupNorm(1, in_channels),  # LayerNorm for conv (groups=1)
            nn.Conv1d(in_channels, mid, 1),
            nn.GELU(),
            nn.Conv1d(mid, out_channels, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return x




class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, activation):
        super().__init__()
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "gelu":
            act_layer = nn.GELU()
        else:
            raise ValueError("Unsupported activation function")

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False, padding_mode="circular"),
            act_layer,
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, bias=False, padding_mode="circular"),
            act_layer,
        )

    def forward(self, x):
        return self.net(x)



class Down(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block1 = ConvNeXtBlock(in_ch)
        self.block2 = ConvNeXtBlockV2(in_ch, in_ch * 2)
        self.norm = nn.GroupNorm(1, in_ch * 2)
        self.downsample = nn.Conv1d(in_ch * 2, in_ch * 2, 2, stride=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        skip = self.norm(x)
        x = self.downsample(skip)

        return skip, x

class Up(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = ConvNeXtBlockV2(in_ch, in_ch // 2)  # concat with skip, so in_ch is doubled
        self.norm = nn.GroupNorm(1, in_ch // 2)
        

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        x = self.norm(x)
        return x



class ConvNeXtUnet(Seq2Seq):
    def __init__(self, lr: float = 1e-3, input_channels:int = 1, channels:int=32, patch_len:int=15, patch_stride:int=7, patch_padding:int=7, depth:int=3):
        super().__init__(lr=lr)
        self.save_hyperparameters()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, patch_len, stride=patch_stride, padding=patch_padding, padding_mode='circular'),  # patchify stem (simplified for 32x32)
            nn.GroupNorm(1, channels))
        #[TODO] explore patching with recovering !!


        self.blks_down = nn.Sequential()
        self.blks_up = nn.Sequential()

        in_ch = channels

        for d in range(depth):
            self.blks_down.add_module("down"+str(d + 1), Down(in_ch))

            in_ch = in_ch * 2
            

        self.bottleneck = ConvNeXtBlockV2(in_ch, in_ch * 2)
        in_ch = in_ch * 2

        for d in range(depth):
            self.blks_up.add_module("up"+str(depth-d), Up(in_ch))
            in_ch //= 2

        
        """
        e.g. depth = 3
        self.down1 = Down(1, 64, kernel_size, padding, activation)
        self.down2 = Down(64, 128, kernel_size, padding, activation)
        self.down3 = Down(128, 256, kernel_size, padding, activation)
        self.bottleneck = Block(256, 512, kernel_size, padding, activation)
        self.up3 = Up(512, 256, kernel_size, padding, activation)
        self.up2 = Up(256, 128, kernel_size, padding, activation)
        self.up1 = Up(128, 64, kernel_size, padding, activation)
        """

        self.head = nn.Conv1d(channels * 2, 1, 1)#1 as kernel size to recover image



    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous() #x=(B,H,1)-> x=(B,1,H)
        x = self.stem(x)  # Apply stem to expand channels x = (B, channels, H)

        skips = []
        for i, down_block in enumerate(self.blks_down):
            skip, x = down_block(x)
            skips.append(skip)

       
        x = self.bottleneck(x)  # x=(B, 256, H/8) -> x=(B, 512, H/8)
        
        for i, up_block in enumerate(self.blks_up):

            skip = skips[-(i+1)]
            x = up_block(x, skip)

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
        return out, None
        return out[:, pastwindow:pastwindow + num_steps], None #No attention weights in UNet [TODO] chage this by convolution masked shown
    
    def predict_loss(self, ref, target, pastwindow):
        """
        ref shape (batch_size, 400, 1)
        target shape (batch_size, 400, 1)

        """
        #past = ref[:, -pastwindow:, :] #shape (batch_size, pastwindow, 1)
        #ref = torch.cat((past, ref), dim=1) #shape (batch_size, pastwindow + 400, 1)

        pred = self(ref) #shape (batch_size, pastwindow + 400, 1)
        loss = self.loss(pred, target, pastwindow) #shape (batch_size,)

        return loss
    
    def get_Jacobian(self):
        """Return a function that computes the Jacobian of predict_loss.

        For the bound method self.predict_loss, argnums=0 refers to ref,
        argnums=1 refers to target, and argnums=2 refers to pastwindow.
        """

        return torch.func.jacrev(self.predict_loss, argnums=0, has_aux=False, chunk_size=None)     
    
if __name__ == "__main__":
    model = ConvNeXtUnet(1e-3, input_channels=1, channels=32, patch_len=7, patch_stride=1, patch_padding=3, depth=3)

    x = torch.randn(2, 600, 1)  #(B, H, 1)

    out = model(x)

    print(x.shape)
    print(out.shape)  #(B, 1, H)

    print(model.depth)

    


    
    
    

    
