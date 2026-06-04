from base_model import HyperParameters
import torch
import wandb
import matplotlib.pyplot as plt


def plot_pred_vs_target(pred, target, ref, pastwindow):
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    ref = ref.squeeze(-1)
    
    shape = ref.shape[0]
    #x_axis_ref = range(0, shape - pastwindow)
    #x_axis_pred = range(shape - pastwindow)


    fig, ax = plt.subplots()
    ax.plot(pred, label=r"$\hat{y}$", linewidth=2)
    ax.plot(target, label=r"$y$", linestyle="--")
    ax.plot(ref, label=r"$x$", linestyle=":", color="green")
    #ymin = min(pred.min().item(), target.min().item(), ref.min().item())
    #ymax = max(pred.max().item(), target.max().item(), ref.max().item())
    #ax.vlines(0, ymin=ymin, ymax=ymax, color='gray', linestyle='--', linewidth=0.5)
    
    #x_min = -pastwindow
    xticks = [0, 100, 200, 300, 400]
    ax.set_xticks(xticks)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$I^{norm}$")
    ax.legend()
    return fig, pred.numpy(), target.numpy(), ref.numpy()

class Trainer(HyperParameters):
    """The base class for training models with data.

    Defined in :numref:`subsec_oo-design-models`"""
    
    def __init__(self, max_epochs, 
                 num_gpus=0, 
                 gradient_clip_val=0, 
                 time_invariance = False,
                 save_model_folder = None, 
                 save_every = 25, 
                 samples_every = 25,
                 project_name = "TFE", 
                 model_name = "model"):
        """Defined in :numref:`sec_use_gpu`"""
        self.save_hyperparameters()
        self.gpus = [torch.device(f'cuda:{i}') for i in range(min(num_gpus, torch.cuda.device_count()))]

    def prepare_data(self, data):
        #Load all dataloaders
        self.batchsize = data.batch_size
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.test_dataloader = data.test_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
        self.num_test_batches = (len(self.test_dataloader)
                                  if self.test_dataloader is not None else 0)
        
        self.pastwindow = data.pastwindow
        self.data_augmented = data.data_augmented

    def prepare_model(self, model):
        #Load model on good device
        """Defined in :numref:`sec_use_gpu`"""
        model.trainer = self
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optim,
            T_0=20,
            T_mult=1,
            eta_min=0.0001,
        )
        #need the scheduler here [TODO]
        self.Wandb_init()
        self.epoch = 0
        self.step = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.past_eval_loss = None
        self.losses_train = torch.zeros(self.samples_every).to(self.gpus[0] if self.gpus else torch.device("cpu"))
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            #self.train_batch_idx = 0
            self.val_batch_idx = 0
            

        if self.save_model_folder is not None:
            torch.save(self.model.state_dict(), f"{self.save_model_folder}/final_model.pt")
        wandb.finish() #Be sure all logs are flushed

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        #Keep mean losses and send to wandb
       
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch), self.pastwindow)
            self.losses_train[self.train_batch_idx] = loss.detach() 
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optim.step()

               
                if self.train_batch_idx == (self.samples_every - 1) :
                    self.scheduler.step()
                    self.train_batch_idx = 0
                    self.model.eval()
                    with torch.no_grad():
                        losses_eval = torch.zeros(self.num_val_batches).to(self.gpus[0] if self.gpus else torch.device("cpu"))
                        first_batch = True

                        
                        table = None
                        for batch in self.val_dataloader:
                            prepared_batch = self.prepare_batch(batch)
                            loss = self.model.validation_step(prepared_batch, self.pastwindow)
                            losses_eval[self.val_batch_idx] = loss
                            self.val_batch_idx += 1

                            if first_batch :#show samples from first batch
                                table = wandb.Table(columns=["sample", "plot", "pred", "target", "ref"])
                                pred_batch = [tensor[:10] for tensor in prepared_batch[:-1]]#take only 10 samples
                                target_batch = prepared_batch[-1][:10]
                                pred, attention_weights = self.model.predict_step(pred_batch, 400, self.pastwindow, save_attention_weights=False)
                                
                                first_batch = False

                                pred = pred.detach().cpu()
                                target_batch = target_batch.detach().cpu()
                                ref_batch = pred_batch[0].detach().cpu() 
                                for i in range(10):
                                    fig, pred_array, target_array, ref_array = plot_pred_vs_target(
                                        pred[i], target_batch[i], ref_batch[i], self.pastwindow
                                    )
                                    table.add_data(i, wandb.Image(fig), pred_array, target_array, ref_array)
                                    plt.close(fig)

                        mean_loss_eval = losses_eval.mean().item() #on 25 batches 12800 datas
                        mean_loss_train = self.losses_train.mean().item() #on all eval: 130000 datas

                        if self.past_eval_loss is None :
                            log_dict = {
                                "step": self.step,
                                "train loss": mean_loss_train,
                            }
                        else:
                            log_dict = {
                                "step": self.step,
                                "train loss": mean_loss_train,
                                "validation loss": self.past_eval_loss,
                            }
                    
                        
                        log_dict[f"samples_step_{self.step}"] = table
                        
                    self.past_eval_loss = mean_loss_eval
                    wandb.log(log_dict)
                    self.model.train()
                    self.val_batch_idx  = 0
                else :
                    self.train_batch_idx += 1

                if self.save_model_folder is not None and self.step % self.save_every == 0:
                    torch.save(self.model.state_dict(), f"{self.save_model_folder}/model_step_{self.step}.pt")

            self.step += 1

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        #batch is a tuple of tensors: batch = [X, Y_target]
        if self.gpus:
            batch = [a.to(self.gpus[0]) for a in batch]
        return batch

    def clip_gradients(self, grad_clip_val, model):
        """Defined in :numref:`sec_rnn-scratch`"""
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm
    
    def Wandb_init(self):
        wandb.init(
            project=self.project_name,
            config={
                "model": self.model_name,
                "batch_size": self.batchsize,
                "dropout": self.model.decoder.dropout,
                "max_epochs": self.max_epochs,
                "learning_rate": self.model.lr,
            }
        )

class UNetTrainer(Trainer):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0, time_invariance = False,save_model_folder=None, save_every=5, samples_every=2, project_name="TFE", model_name="UNet"):
        super().__init__(max_epochs, num_gpus, gradient_clip_val, time_invariance,save_model_folder, save_every, samples_every, project_name, model_name)

    def Wandb_init(self):
        wandb.init(
            project=self.project_name,
            config={
                "model": self.model_name,
                "batch_size": self.batchsize,
                "depth": self.model.depth,
                "wide": self.model.wide,
                "activation": self.model.activation,
                "kernel_size": self.model.kernel_size,
                "padding": self.model.padding,
                "max_epochs": self.max_epochs,
                "learning_rate": self.model.lr,
                "past window": self.pastwindow,
                "data_augm": self.data_augmented,
                "gradient_clip": self.gradient_clip_val,
                "time_invariance": self.time_invariance,
            }
        )
    
    def prepare_batch(self, batch):
        #X shape (batch_size, 400, 1)
        #Y_target shape (batch_size, 400, 1)

        batch = super().prepare_batch(batch) # [X, Y_target]
        
        idx = torch.randint(0, batch[0].shape[1], (1,)).item()  # integer in [0, 399]

        if(len(batch)==1):
            X = batch[0]
            if self.time_invariance:
                X = torch.roll(X, shifts=idx, dims=1)
           

            #past_window = X[:, -self.pastwindow:, :] #shape (batch_size, pastwindow, 1)
            #X = torch.cat((past_window, X), dim=1) #shape (batch_size, pastwindow + 400, 1)
            batch = (X,)
            return batch
        
        # Shifts
        X, Y_target = batch
        if self.time_invariance:
            Y_target = torch.roll(Y_target, shifts=idx, dims=1)
            X = torch.roll(X, shifts=idx, dims=1)
        

        #past_window = X[:, -self.pastwindow:, :] #shape (batch_size, pastwindow, 1)
        #X = torch.cat((past_window, X), dim=1) #shape (batch_size, pastwindow + 400, 1)

        #[TODO] if removed data augmented, change in wandb init
        

        return X, Y_target

class UNetTrainerConvNeXt(UNetTrainer):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0, time_invariance = False,save_model_folder=None, save_every=5, samples_every=2, project_name="TFE", model_name="UNetConvNeXt"):
        super().__init__(max_epochs, num_gpus, gradient_clip_val, time_invariance,save_model_folder, save_every, samples_every, project_name, model_name)

    def Wandb_init(self):
        wandb.init(
            project=self.project_name,
            config={
                "model": self.model_name,
                "batch_size": self.batchsize,
                "depth": self.model.depth,
                "patch_len": self.model.patch_len,
                "patch_stride": self.model.patch_stride,
                "patch_padding": self.model.patch_padding,
                "wide": self.model.channels,
                "max_epochs": self.max_epochs,
                "learning_rate": self.model.lr,
                "past window": self.pastwindow,
                "data_augm": self.data_augmented,
                "gradient_clip": self.gradient_clip_val,
                "time_invariance": self.time_invariance,
            }
        )
if __name__ == "__main__":

    print("Nothing implemented")
