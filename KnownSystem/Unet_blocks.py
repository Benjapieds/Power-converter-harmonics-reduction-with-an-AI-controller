import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import Seq2Seq

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
    def __init__(self, in_ch, out_ch, kernel_size, padding, activation):
        super().__init__()
        self.block = Block(in_ch, out_ch, kernel_size, padding, activation)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        skip = self.block(x)
        return skip, self.pool(skip)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, activation):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = Block(out_ch * 2, out_ch, kernel_size, padding, activation)  # concat with skip

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class UNet(Seq2Seq):
    def __init__(self, lr, kernel_size=3, padding=1, depth = 3, wide = 64, activation = "relu"):
        super().__init__(lr=lr)
        self.save_hyperparameters()

        self.blks_down = nn.Sequential()
        self.blks_up = nn.Sequential()

        in_ch = 1
        out_ch = wide

        for d in range(depth):
            self.blks_down.add_module("down"+str(d + 1), Down(in_ch, out_ch, kernel_size, padding, activation))

            in_ch = out_ch
            out_ch *= 2

        self.bottleneck = Block(in_ch, out_ch, kernel_size, padding, activation)

        for d in range(depth):
            self.blks_up.add_module("up"+str(depth-d), Up(out_ch, in_ch, kernel_size, padding, activation))

            out_ch = in_ch
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

        self.head = nn.Conv1d(wide, 1, 1)#1 as kernel size to recover image

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous() #x=(B,H,1)-> x=(B,1,H)

        skips = []
        for i, down_block in enumerate(self.blks_down):
            skip, x = down_block(x)
            skips.append(skip)

        # down path (e.g depth=3)
        # x=(B, 1, H) -> s1=(B, 64, H), x=(B, 64, H/2)
        # x=(B, 64, H/2) -> s2=(B, 128, H/2), x=(B, 128, H/4)
        # x=(B, 128, H/4) -> s3=(B, 256, H/4), x=(B, 256, H/8)
             
        x = self.bottleneck(x)  # x=(B, 256, H/8) -> x=(B, 512, H/8)
        
        for i, up_block in enumerate(self.blks_up):

            skip = skips[-(i+1)]
            x = up_block(x, skip)
            
        
        # up path (e.g depth=3)
        # x=(B, 512, H/8), s3=(B, 256, H/4) -> x=(B, 256, H/4)
        # x=(B, 256, H/4), s2=(B, 128, H/2) -> x=(B, 128, H/2)
        # x=(B, 128, H/2), s1=(B, 64, H) -> x=(B, 64, H)
          
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

        return out[:, pastwindow:pastwindow + num_steps], None #No attention weights in UNet [TODO] chage this by convolution masked shown
    
    def predict_loss(self, ref, target, pastwindow):
        """
        ref shape (batch_size, 400, 1)
        target shape (batch_size, 400, 1)

        """
        past = ref[:, -pastwindow:, :] #shape (batch_size, pastwindow, 1)
        ref = torch.cat((past, ref), dim=1) #shape (batch_size, pastwindow + 400, 1)

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
    model = UNet(1e-3, kernel_size=3, padding=1, depth=3, wide=64, activation="gelu")

    x = torch.randn(2, 600, 1)  #(B, H, 1)

    out = model(x)

    print(x.shape)
    print(out.shape)  #(B, 1, H)

    print(model.depth)

    


    
    
    
