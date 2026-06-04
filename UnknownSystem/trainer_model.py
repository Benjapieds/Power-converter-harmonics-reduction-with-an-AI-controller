from base_model import HyperParameters
import torch
import wandb
import matplotlib.pyplot as plt

#[TODO] correct loss evaluation
def plot_pred_vs_target(pred, target, ref, pastwindow, past):
    if past is None:
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        ref = ref.squeeze(-1)
        past = past.squeeze(-1) #shape (N-1, 2, 400)
        Xpast = past[:,0,:] #shape (N-1, 400)
        Ypast = past[:,1,:] #shape (N-1, 400)
        
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

    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    ref = ref.squeeze(-1)
    past = past.squeeze(-1) #shape (N-1, 2, 400)
    Xpast = past[:,0,:] #shape (N-1, 400)
    Ypast = past[:,1,:] #shape (N-1, 400)
    
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

    num_context_plots = 4
    fig2, axes = plt.subplots(num_context_plots, 1, figsize=(8, 5), sharex=True)
    if num_context_plots == 1:
        axes = [axes]

    num_available_contexts = min(num_context_plots, Xpast.shape[0])
    for i, ax2 in enumerate(axes):
        if i >= num_available_contexts:
            ax2.axis("off")
            continue

        ax2.plot(Xpast[i], label=r"$x^{(i)}$", linestyle="--")
        ax2.plot(Ypast[i], label=r"$y^{(i)}$")
        ax2.set_xticks([0, 100, 200, 300, 400])
        ax2.set_ylabel(r"$I^{norm}$")
        if i < num_context_plots - 1:
            ax2.tick_params(labelbottom=False)
        else:
            ax2.set_xlabel(r"$t$")
        if i == 0:
            ax2.legend()




    
    return fig, fig2, pred.numpy(), target.numpy(), ref.numpy()


class HypernetTrainer(HyperParameters):
    def __init__(self,
                max_epochs,
                num_gpus=0,
                gradient_clip_val=0,
                time_invariance = False,
                save_model_folder=None,
                save_every=5, 
                samples_every=2, 
                project_name="TFE",
                model_name="Hypernet"):
        
        self.save_hyperparameters()
        self.gpus = [torch.device(f'cuda:{i}') for i in range(min(num_gpus, torch.cuda.device_count()))]

    def prepare_data(self, data):
        #Load all dataloaders
        self.batchsize = data.batch_size
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        #self.test_dataloader = data.test_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
        # self.num_test_batches = (len(self.test_dataloader)
        #                           if self.test_dataloader is not None else 0)
        
        self.pastwindow = data.past_window
        self.data_augmented = False
        self.pastsize= data.trajectory_dim
        self.norm = data.max_value.item()
    
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
            T_0=50,
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
            #print("Before Training step", flush=True)
            loss = self.model.training_step(self.prepare_batch(batch), self.pastwindow)
            #print("After training step", flush=True)
            self.losses_train[self.train_batch_idx] = loss.detach() 
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                #print("After loss backward", flush=True)
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optim.step()

               
                if self.train_batch_idx == (self.samples_every - 1):
                    self.scheduler.step()
                    self.train_batch_idx = 0
                    self.model.eval()
                    with torch.no_grad():
                        losses_eval = torch.zeros(self.num_val_batches).to(self.gpus[0] if self.gpus else torch.device("cpu"))
                        first_batch = True

                        
                        table = None
                        #print("Before Eval", flush=True)
                        for batch in self.val_dataloader:
                            #print("In batch Eval", flush=True)
                            prepared_batch = self.prepare_batch(batch)
                            loss = self.model.validation_step(prepared_batch, self.pastwindow)
                            losses_eval[self.val_batch_idx] = loss
                            self.val_batch_idx += 1

                            if first_batch :#show samples from first batch
                                #print("Before Table", flush=True)
                                table = wandb.Table(columns=["sample", "plot1", "plot2", "pred", "target", "ref"])
                                pred_batch = [tensor[:10] for tensor in prepared_batch[:-1]]#take only 10 samples, [x, past]
                                target_batch = prepared_batch[-1][:10] # [y]
                                pred, attention_weights = self.model.predict_step(pred_batch, 400, self.pastwindow, save_attention_weights=False)
                                
                                first_batch = False

                                pred = pred.detach().cpu()
                                target_batch = target_batch.detach().cpu()
                                ref_batch = pred_batch[0].detach().cpu()
                                past_batch = pred_batch[1].detach().cpu() #shape (10, N-1, 2, 400, 1)
                                for i in range(10):
                                    fig, fig2, pred_array, target_array, ref_array = plot_pred_vs_target(
                                        pred[i], target_batch[i], ref_batch[i], self.pastwindow, past_batch[i]
                                    )
                                    table.add_data(i, wandb.Image(fig), wandb.Image(fig2), pred_array, target_array, ref_array)
                                    plt.close(fig)
                                    plt.close(fig2)
                                #print("After Table", flush=True)

                        mean_loss_eval = losses_eval.mean().item() #on 25 batches 12800 datas
                        mean_loss_train = self.losses_train.mean().item() #on all eval: 130000 datas

                        #print("After Mean Stat", flush=True)


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
                    
                        #print("Before Logging", flush=True)
                        log_dict[f"samples_step_{self.step}"] = table
                    
                    #print("Before SendLog", flush=True)
                    self.past_eval_loss = mean_loss_eval
                    wandb.log(log_dict)
                    self.model.train()
                    self.val_batch_idx  = 0
                    #print("After SendLog", flush=True)
                else :
                    self.train_batch_idx += 1

                if self.save_model_folder is not None and self.step % self.save_every == 0:
                    torch.save(self.model.state_dict(), f"{self.save_model_folder}/model_step_{self.step}.pt")

            self.step += 1

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        """
        batch is tensor of shape (batch_size, N, 2, 400)

        return a batch on GPU with shape:
        liste of [ x[B, 400+pastwindow, 1], past[B, N-1, 2, 400, 1],  y[B, 400, 1]] 
        
        """
        if self.gpus:
            batch = batch.to(self.gpus[0]).unsqueeze(-1) #shape (B, N, 2, T, 1)
        else:
            batch = batch.unsqueeze(-1) #shape (B, N, 2, T, 1)
        batch = batch / self.norm #normalize the batch
        past = batch[:, :-1, :,:,:].contiguous().float() # (B, N-1, 2, T, 1)
        current = batch[:, -1, :, :, :].contiguous().float() # (B, 2, T, 1)
        x = current[:,0, :, :].contiguous().float() # (B, T, 1)
        y = current[:,1,:,:].contiguous().float() # (B, T, 1)

        #add the past_window
        #past_window = x[:, -self.pastwindow:, :] #shape (batch_size, pastwindow, 1)
        #x = torch.cat((past_window, x), dim=1) #shape (batch_size, pastwindow + 400, 1)
        batch = (x, past, y)

        return batch

    
    def Wandb_init(self):
        wandb.init(
            project=self.project_name,
            config={
                "model": self.model_name, #Trainer params
                "lr": self.model.lr, #Optimizer params
                "past_size": self.pastsize,# Data params
                "batch_size": self.batchsize,
                "max_epochs": self.max_epochs,
                "learning_rate": self.model.lr,
                "gradient_clip": self.gradient_clip_val,
                "past window": self.pastwindow,
                "time_invariance": self.time_invariance,#Unet params
                "depth": self.model.depth,
                "wide": self.model.wide,
                "activation": self.model.activation,
                "kernel_size": self.model.kernel_size,
                "padding": self.model.padding,
                "channels": self.model.channels,#Deepset params
                "patch_len": self.model.patch_len,
                "patch_stride": self.model.patch_stride,
                "patch_padding": self.model.patch_padding,
                "depth_deepset": self.model.depth_deepset,
                "aggregation": self.model.aggregation,
                "input_dim": self.model.input_dim,
            }
        )

class HypernetTrainerConvNeXt(HypernetTrainer):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0, time_invariance = False,save_model_folder=None, save_every=5, samples_every=2, project_name="TFE", model_name="HypernetConvNeXt"):
        super().__init__(max_epochs, num_gpus, gradient_clip_val, time_invariance,save_model_folder, save_every, samples_every, project_name, model_name)

    def Wandb_init(self):
        wandb.init(
            project=self.project_name,
            config={
                "model": self.model_name, #Trainer params
                "lr": self.model.lr, #Optimizer params
                "past_size": self.pastsize,# Data params
                "batch_size": self.batchsize,
                "max_epochs": self.max_epochs,
                "learning_rate": self.model.lr,
                "gradient_clip": self.gradient_clip_val,
                "past window": self.pastwindow,
                "time_invariance": self.time_invariance,#Unet params
                "depth": self.model.depth,
                "channels": self.model.channels,
                "patch_len": self.model.patch_len,
                "patch_padding": self.model.patch_padding,
                "patch_stride": self.model.patch_stride,
                
                "D_channels": self.model.channels_Deepset,#Deepset params
                "D_patch_len": self.model.patch_len_Deepset,
                "D_patch_stride": self.model.patch_stride_Deepset,
                "D_patch_padding": self.model.patch_padding_Deepset,
                "D_deepset": self.model.depth_Deepset,
                "aggregation": self.model.aggregation,
                "input_dim": self.model.input_dim,
                "factor": self.model.factor,
            }
        )

class ClassicalTrainerConvNeXt(HypernetTrainer):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0, time_invariance = False,save_model_folder=None, save_every=5, samples_every=2, project_name="TFE", model_name="ClassicalConvNeXt"):
        super().__init__(max_epochs, num_gpus, gradient_clip_val, time_invariance,save_model_folder, save_every, samples_every, project_name, model_name)

    def Wandb_init(self):
        wandb.init(
            project=self.project_name,
            config={
                "model": self.model_name, #Trainer params
                
            }
        )
    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        """
        batch is tensor of shape (batch_size, N, 2, 400)

        return a batch on GPU with shape:
        liste of [ x[B, 400+pastwindow, 1], past[B, N-1, 2, 400, 1],  y[B, 400, 1]] 
        
        """
        if self.gpus:
            batch = batch.to(self.gpus[0]).unsqueeze(-1) #shape (B, N, 2, T, 1)
        else:
            batch = batch.unsqueeze(-1) #shape (B, N, 2, T, 1)
        batch = batch / self.norm #normalize the batch
        past = batch[:, :-1, :,:,:].contiguous().float() # (B, N-1, 2, T, 1)
        current = batch[:, -1, :, :, :].contiguous().float() # (B, 2, T, 1)
        x = current[:,0, :, :].contiguous().float() # (B, T, 1)
        y = current[:,1,:,:].contiguous().float() # (B, T, 1)

        #add the past_window
        #past_window = x[:, -self.pastwindow:, :] #shape (batch_size, pastwindow, 1)
        #x = torch.cat((past_window, x), dim=1) #shape (batch_size, pastwindow + 400, 1)
        batch = (x, y)

        return batch
    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        #Keep mean losses and send to wandb
        
        self.model.train()
        for batch in self.train_dataloader:
            #print("Before Training step", flush=True)
            loss = self.model.training_step(self.prepare_batch(batch), self.pastwindow)
            #print("After training step", flush=True)
            self.losses_train[self.train_batch_idx] = loss.detach() 
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                #print("After loss backward", flush=True)
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optim.step()

               
                if self.train_batch_idx == (self.samples_every - 1):
                    self.scheduler.step()
                    self.train_batch_idx = 0
                    self.model.eval()
                    with torch.no_grad():
                        losses_eval = torch.zeros(self.num_val_batches).to(self.gpus[0] if self.gpus else torch.device("cpu"))
                        first_batch = True

                        
                        table = None
                        #print("Before Eval", flush=True)
                        for batch in self.val_dataloader:
                            #print("In batch Eval", flush=True)
                            prepared_batch = self.prepare_batch(batch)
                            loss = self.model.validation_step(prepared_batch, self.pastwindow)
                            losses_eval[self.val_batch_idx] = loss
                            self.val_batch_idx += 1

                            if first_batch :#show samples from first batch
                                #print("Before Table", flush=True)
                                table = wandb.Table(columns=["sample", "plot1", "pred", "target", "ref"])
                                pred_batch = [tensor[:10] for tensor in prepared_batch[:-1]]#take only 10 samples, [x, past]
                                target_batch = prepared_batch[-1][:10] # [y]
                                pred, attention_weights = self.model.predict_step(pred_batch, 400, self.pastwindow, save_attention_weights=False)
                                
                                first_batch = False

                                pred = pred.detach().cpu()
                                target_batch = target_batch.detach().cpu()
                                ref_batch = pred_batch[0].detach().cpu()
                                #past_batch = pred_batch[1].detach().cpu() #shape (10, N-1, 2, 400, 1)
                                for i in range(10):
                                    fig,  pred_array, target_array, ref_array = plot_pred_vs_target(
                                        pred[i], target_batch[i], ref_batch[i], self.pastwindow, None
                                    )
                                    table.add_data(i, wandb.Image(fig), pred_array, target_array, ref_array)
                                    plt.close(fig)
                                    
                                #print("After Table", flush=True)

                        mean_loss_eval = losses_eval.mean().item() #on 25 batches 12800 datas
                        mean_loss_train = self.losses_train.mean().item() #on all eval: 130000 datas

                        #print("After Mean Stat", flush=True)


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
                    
                        #print("Before Logging", flush=True)
                        log_dict[f"samples_step_{self.step}"] = table
                    
                    #print("Before SendLog", flush=True)
                    self.past_eval_loss = mean_loss_eval
                    wandb.log(log_dict)
                    self.model.train()
                    self.val_batch_idx  = 0
                    #print("After SendLog", flush=True)
                else :
                    self.train_batch_idx += 1

                if self.save_model_folder is not None and self.step % self.save_every == 0:
                    torch.save(self.model.state_dict(), f"{self.save_model_folder}/model_step_{self.step}.pt")

            self.step += 1

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
        
        self.pastwindow = data.past_window
        self.data_augmented = False

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
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            

        if self.save_model_folder is not None:
            torch.save(self.model.state_dict(), f"{self.save_model_folder}/final_model.pt")
        wandb.finish() #Be sure all logs are flushed

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        #Keep mean losses and send to wandb
        losses_train = torch.zeros(self.samples_every).to(self.gpus[0] if self.gpus else torch.device("cpu"))
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch), self.pastwindow)
            losses_train[self.train_batch_idx] = loss.detach() 
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optim.step()

               
                if self.train_batch_idx == (self.samples_every - 1):
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
                                table = wandb.Table(columns=["sample", "plot1", "plot2", "pred", "target", "ref"])
                                pred_batch = [tensor[:10] for tensor in prepared_batch[:-1]]#take only 10 samples
                                target_batch = prepared_batch[-1][:10]
                                pred, attention_weights = self.model.predict_step(pred_batch, 400, self.pastwindow, save_attention_weights=False)
                                
                                first_batch = False

                                pred = pred.detach().cpu()
                                target_batch = target_batch.detach().cpu()
                                ref_batch = pred_batch[0].detach().cpu() 
                                for i in range(10):
                                    fig, fig2, pred_array, target_array, ref_array = plot_pred_vs_target(
                                        pred[i], target_batch[i], ref_batch[i], self.pastwindow
                                    )
                                    table.add_data(i, wandb.Image(fig), wandb.Image(fig2), pred_array, target_array, ref_array)
                                    plt.close(fig)
                                    plt.close(fig2)

                        mean_loss_eval = losses_eval.mean().item() #on 25 batches 12800 datas
                        mean_loss_train = losses_train.mean().item() #on all eval: 130000 datas

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
           

            past_window = X[:, -self.pastwindow:, :] #shape (batch_size, pastwindow, 1)
            X = torch.cat((past_window, X), dim=1) #shape (batch_size, pastwindow + 400, 1)
            batch = (X,)
            return batch
        
        # Shifts
        X, Y_target = batch
        if self.time_invariance:
            Y_target = torch.roll(Y_target, shifts=idx, dims=1)
            X = torch.roll(X, shifts=idx, dims=1)
        

        past_window = X[:, -self.pastwindow:, :] #shape (batch_size, pastwindow, 1)
        X = torch.cat((past_window, X), dim=1) #shape (batch_size, pastwindow + 400, 1)

        #[TODO] if removed data augmented, change in wandb init
        

        return X, Y_target


if __name__ == "__main__":

    print("Nothing implemented")
