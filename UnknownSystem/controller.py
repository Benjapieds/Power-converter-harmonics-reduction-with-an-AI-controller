import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.optimize import minimize


from deepset_convNeXt import HyperNetwork
from data_model import FullData


import Plant_torch
import baseline

#Compute the THD of a signal define over 2 periods[TODO evaluate it in torch]
def THD(signal, sampling_period):
    if sampling_period != 1e-4:
        raise Exception("THD must me computed for 10kHz sampling")
    
    fourier = np.fft.rfft(signal)
    
    #As 10kHz signal, freq 2 = 50Hz, freq 4 = 100 Hz etc
    harmonics = np.arange(1, 52) #up to 50th harmonic
   
    harmonics_values = np.zeros(fourier.size, dtype=np.complex64)
    Voltages_RMS = np.zeros_like(harmonics, dtype=np.float16)

    for i in harmonics:
        harmonics_values[i] = fourier[i * 2]

        voltage_harmonics = np.fft.irfft(harmonics_values)
        harmonics_values[i] = 0  #reset for next iteration

        Voltages_RMS[i-1] = np.max(voltage_harmonics) / np.sqrt(2)

    THD = np.sqrt(np.sum(Voltages_RMS[1:]**2)) / Voltages_RMS[0]
    

    return THD

def MSE(signal1, signal2):
    if signal1.shape != signal2.shape:
        raise Exception("Signals must have the same shape")
    
    return np.mean((signal1 - signal2)**2)



def Load_model(model, path):
    model.load_state_dict(torch.load(path))

def Optimizer_f(model, target, past_window, past_context):
    """
    Return f(x) -> float with f(x) = lambda x: model.predict_loss(x, target, past_window, past_context)

    target: torch tensor of shape (1, 400, 1)
    past_window: int, the number of past time steps used as input to the model
    past_context: torch tensor of shape (1, past_context_dim, 2, 400, 1) the context used as input to the model for prediction

    Take care of good shape and types for torch and numpy compatibility.
    
    """
    def f(x):
        #x is a numpy array of shape (400)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(model.device) #shape (1, 400, 1)

        loss = model.predict_loss(x_tensor, target, past_window, past_context) #shape (1,)
        return loss.item() #return a scalar

    return f

def Optimizer_J(model, target, past_window, past_context):
    """
    Return J(x) -> numpy array with J(x) = lambda x: model.get_Jacobian()(x, target, past_window, past_context)

    target: torch tensor of shape (1, 400, 1)
    past_window: int, the number of past time steps used as input to the model
    past_context: torch tensor of shape (1, past_context_dim, 2, 400, 1) the context used as input to the model for prediction

    Take care of good shape and types for torch and numpy compatibility.
    
    """
    jacobian_gn = model.get_Jacobian()

    def J(x):
        # x is a numpy array of shape (400)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(model.device) #shape (1, 400, 1)
        jacobian = jacobian_gn(x, target, past_window, past_context) #shape (1, 400, 1)
        return jacobian.squeeze(0).squeeze(-1).cpu().numpy() #shape (400,)

    return J
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    
   
    

    from variability import I_refs, Kps, Kis, t_Rs, Ls
    from matplotlib_init import init_matplotlib
    init_matplotlib()

    batch_size = 1

    

    Rs_up = 7.0
    Rs_down = 1.0

    # Rs_up = 7.0
    # Rs_down = 4.0

    t_Rs = 0.0158362
    ds = 0.0103472
    # Rs_up = 11.0
    # Rs_down = 3.5
    # t_Rs = 0.0078362
    # ds = 0.0083472
    print(f"Rs_up: {Rs_up}, Rs_down: {Rs_down}, t_Rs: {t_Rs}, d: {ds}")



    #batch_size = 2
   
    Iref = torch.as_tensor(I_refs, dtype=torch.float32).flatten().to(device).repeat(batch_size) # shape (10000)
    Kp = torch.tensor([Kps], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    Ki = torch.tensor([Kis], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    R_down = torch.tensor([Rs_down], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    R_up = torch.tensor([Rs_up], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    t_R = torch.tensor([t_Rs], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    d_R = torch.tensor([ds], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    L = torch.tensor([Ls], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)

    input_sequence = Plant_torch.generate_input_sequence_torch(Iref).to(device) #shape (10000)
    initialisation = Plant_torch.generate_initialization_torch(batch_size).to(device)
    print(f"input_sequence shape: {input_sequence.shape}")
    print(f"Initialisation shape: {initialisation.shape}")
    controller_params = Plant_torch.generate_controller_params_torch(Kp, Ki)
    params_bis = Plant_torch.generate_Vdc_bool_Transducer_torch(batch_size).to(device)
    print(f"Params bis shape: {params_bis.shape}")
    params =  baseline.generate_Rs_Ls_Es_torch(R_down, R_up, t_R, d_R, L)
    print(f"Params shape: {params.shape}")

    

    #Generate with default reference
    output_sequence, final_initialisation, final_controller_params = Plant_torch.system_torch(input_sequence, params, params_bis, controller_params, initialisation)#shape (B, T)
    error = Plant_torch.Error(output_sequence, Iref, shift=True)#shape (B, T)






    
    THD_sinus = THD(output_sequence.squeeze(0).cpu().numpy(), sampling_period=1e-4)
    print(f"THD of the simulator response to sinus: {THD_sinus}")

    output_sequence =  Plant_torch.Interpolate(output_sequence[:,200:])
    MSE_sinus = MSE(output_sequence.squeeze(0).cpu().numpy(), input_sequence.squeeze(0).cpu().numpy())
    print(f"MSE of the simulator response to sinus: {MSE_sinus}")

    plt.plot(output_sequence.squeeze(0).cpu().numpy() * 10.0, label=r"$y^{sin}$")
    plt.plot(input_sequence.squeeze(0).cpu().numpy() * 10.0, label=r"$x^{sin}$")
    # Legend-only entry
    plt.plot([], [], ' ', label=r"THD = {:.2f}".format(THD_sinus * 100) + "%")
    plt.plot([], [], ' ', label=r"MSE = {:.2e}".format(MSE_sinus))
    plt.xlabel(r"$t$")
    plt.ylabel(r"$[A]$")
    plt.legend()
    plt.savefig("response_to_sinus.svg")
    plt.show()
    #plt.close()


    #Generate with default reference
    
        
    batch_kp = torch.tensor([0.0, 0.1, -0.5, 1.3, 0.5], dtype=torch.float32).to(device)
    batch_ki = torch.tensor([0.0, 0.01, -0.02, 0.03, 0.0], dtype=torch.float32).to(device)
    batch_window = torch.tensor([0, 4, 2, 3, 1], dtype=torch.int64).to(device)



    sample_Kp = np.linspace(0.0, 1.5, 25)
    choice = []
    for i in range(5):
        seed = 32 + i
        rg = np.random.default_rng(seed)

        if i == 0 :
              continue
        elif i == 1 :
            selected_combination1 = rg.choice(sample_Kp, size=25, replace=False)  # Select 10 random combinations
            selected_combination = np.stack((selected_combination1), axis=0)
            choice.append(selected_combination)
    
        elif i == 2 :
            selected_combinations1 = rg.choice(sample_Kp, size=25, replace=False)  # Select 10 random combinations
            selected_combinations2 = rg.choice(sample_Kp, size=25, replace=False)  # Select 10 random combinations
            selected_combination = np.stack((selected_combinations1, selected_combinations2), axis=0)
            choice.append(selected_combination)
        elif i == 3 :
            selected_combinations1 = rg.choice(sample_Kp, size=25, replace=False)  # Select 10 random combinations
            selected_combinations2 = rg.choice(sample_Kp, size=25, replace=False)  # Select 10 random combinations
            selected_combinations3 = rg.choice(sample_Kp, size=25, replace=False)  # Select 10 random combinations
            selected_combination = np.stack((selected_combinations1, selected_combinations2, selected_combinations3), axis=0)
            choice.append(selected_combination)
        elif i == 4 :
            selected_combinations1 = rg.choice(sample_Kp, size=25, replace=False)  # Select 10 random combinations
            selected_combinations2 = rg.choice(sample_Kp, size=25, replace=False)  # Select 10 random combinations
            selected_combinations3 = rg.choice(sample_Kp, size=25, replace=False)  # Select 10 random combinations
            selected_combinations4 = rg.choice(sample_Kp, size=25, replace=False)  # Select 10 random combinations
            selected_combination = np.stack((selected_combinations1, selected_combinations2, selected_combinations3, selected_combinations4), axis=0)
            choice.append(selected_combination)

        print(f"Selected combinations shape: {selected_combination.shape}")
        print(f"Selected combinations: {selected_combination}")

    batch = 5
    er = error.repeat(batch, 1) #shape (5, T)
    b_kp = batch_kp #shape (5,)
    b_ki = batch_ki #shape (5,)
    b_window = batch_window #shape (5,)

    pa = params.repeat(batch, 1, 1) #shape (5, 3, Clock_step)
    pa_bis = params_bis.repeat(batch, 1) #shape (5, param_bis_dim)
    f_c = final_controller_params.repeat(batch, 1) #shape (5, controller_param_dim)
    f_i = final_initialisation.repeat(batch, 1) #shape (5, initialisation_dim)

    input_sequence_bis = input_sequence + Plant_torch.reference_generator(er, b_kp, b_ki, b_window)
    output_sequence, f_i, f_c = Plant_torch.system_torch(input_sequence_bis, pa, pa_bis, f_c, f_i)#shape (B, T)

    output_sequence = Plant_torch.Interpolate(output_sequence[:,200:])
    #Plot all elements in past_context, one line per context index
    #create past_window of shape 1,5,2,400,1
    past_context = torch.stack([
            torch.stack([input_sequence_bis[j], output_sequence[j]], dim=0)
            for j in range(batch)
    ], dim=0).unsqueeze(0).unsqueeze(-1)  # shape (1, batch, 2, 400, 1)

    print(f"Past context shape: {past_context.shape}")







    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    for j in range(batch):
        context_j = past_context[0, j]  # shape (2, 400, 1)
        x_context = context_j[0].detach().cpu().numpy().squeeze(-1) * 10.0
        y_context = context_j[1].detach().cpu().numpy().squeeze(-1) * 10.0

        ax1.plot(x_context, label=rf"$x^{{({j})}}$", linewidth=2)
        ax2.plot(y_context, label=rf"$y^{{({j})}}$", linewidth=2)

    ax1.set_ylabel(r"$[A]$")
    ax1.legend()
    #ax1.grid(True, alpha=0.3)

    ax2.set_ylabel(r"$[A]$")
    ax2.set_xlabel(r"$t$")
    ax2.legend()
    #ax2.grid(True, alpha=0.3)

    #plt.tight_layout()
    plt.savefig(f"past.svg")
    plt.show()

    x_0 = past_context[0,-1,0,:,0].detach().cpu().numpy() #shape (400,)
    y_0 = past_context[0,-1,1,:,0].detach().cpu().numpy() #shape (400,)
    target = past_context[:,0,0,:,:]
    plt.plot(x_0 * 10.0, label=r"$x^{(0)}$")
    plt.plot(y_0 * 10.0, label=r"$y^{(0)}$")
    plt.ylabel(r"$[A]$")
    plt.xlabel(r"$t$")
    plt.legend()
    plt.savefig(f"x0y0.svg")
    plt.show()



    for n in range(5):
        if n == 0:
            continue
        batch = 25 * (n)
        comb_Kp = choice[n-1]
        b_kp = torch.tensor(np.asarray(comb_Kp).reshape(-1), dtype=torch.float32).to(device)
        print(b_kp.shape)
        b_ki = torch.zeros_like(b_kp).to(device)
        b_window = torch.zeros_like(b_kp, dtype=torch.int64).to(device)

        pa = params.repeat(batch, 1, 1) #shape (5, 3, Clock_step)
        pa_bis = params_bis.repeat(batch, 1) #shape (5, param_bis_dim)
        f_c = final_controller_params.repeat(batch, 1) #shape (5, controller_param_dim)
        f_i = final_initialisation.repeat(batch, 1) #shape (5, initialisation_dim)
        er = error.repeat(batch, 1) #shape (5, T)

        input_sequence_bis = input_sequence + Plant_torch.reference_generator(er, b_kp, b_ki, b_window)
        output_sequence, f_i, f_c = Plant_torch.system_torch(input_sequence_bis, pa, pa_bis, f_c, f_i)#shape (B, T)



        output_sequence = Plant_torch.Interpolate(output_sequence[:,200:])

        pt_context = torch.stack([
            torch.stack([
                torch.stack([input_sequence_bis[j], output_sequence[j]], dim=0)
                for j in range(i * n, (i + 1) * n)
            ], dim=0)
            for i in range(25)
        ], dim=0).unsqueeze(-1)  # shape (25, n, 2, 400, 1)

        print(f"Pt context shape: {pt_context.shape}")


        #Define y_target
        normalisation = 1.3583984375
        past_window = 48
        freq = 20e3 #20kHz
        h = 1. / freq
        T = 20e-3 #20ms period
                
        t_ref = torch.arange(0, T, h )
        target = 0.1 * I_refs * torch.sqrt(torch.tensor(2.)) * torch.sin(2 * torch.pi * 50 * t_ref)
        target = target / normalisation #Emulator target
        target = target.unsqueeze(0).unsqueeze(-1) #shape (1, 400, 1)
        path = r"COMPLETE"
    
        model = HyperNetwork(lr= 1e-3,
                                input_channels=1,
                                channels=16,
                                patch_len=7,
                                patch_stride=1,
                                patch_padding=3,
                                depth=2,
                                node_dim=(2, 400, 1),
                                channels_Deepset=32,
                                patch_len_Deepset=7,
                                patch_stride_Deepset=7,
                                patch_padding_Deepset=3,
                                depth_Deepset=2,
                                aggregation="max",
                                input_dim=400,
                                factor=10)
        
        Load_model(model, path)
        model.to(device)
        model.eval()
        target = target.to(device)
        pt_context = pt_context.to(device)
        pt_context = pt_context / normalisation
        #pt_context = torch.zeros_like(pt_context) #shape (1, 4, 2, 400, 1), zero out pt context to evaluate the model response to x0 alone

        #x_0 = past_context[0,-1,0,:,0].detach().cpu().numpy() #shape (400,)
        x_0_torch = torch.tensor(x_0, dtype=torch.float32)
        x_0_torch = x_0_torch / normalisation
        x_0_torch = x_0_torch.to(device)
        x_0_torch = x_0_torch.unsqueeze(0).unsqueeze(-1) #shape (1, 400, 1)
        with torch.no_grad():
                        print(f"Model has {model.count_parameters()}", flush =True)
                        print("#########################", flush=True)

                        
                        # hat_y = model(target, past_context) #shape (1, 440, 1)
                        # plt.plot(target.squeeze(0).squeeze(-1).cpu().numpy() * normalisation * 10.0, label=r"$x^{sin}$")
                        # plt.plot(hat_y.squeeze(0).squeeze(-1).cpu().numpy() * normalisation * 10.0, label=r"$\hat{y}^{sin}$")
                        # plt.ylabel(r"$[A]$")
                        # plt.xlabel(r"$t$")
                        # plt.legend()
                        # plt.savefig(f"Hypernet_response_to_sinus_shifted_model.svg")
                        # plt.show()

                        # hat_y = model(x_0_torch, past_context) #shape (1, 440, 1)
                        # plt.plot(x_0_torch.squeeze(0).squeeze(-1).cpu().numpy() * normalisation * 10.0, label=r"$x^{(0)}$")
                        # plt.plot(hat_y.squeeze(0).squeeze(-1).cpu().numpy() * normalisation * 10.0, label=r"$\hat{y}^{(0)}$")
                        # plt.ylabel(r"$[A]$")
                        # plt.xlabel(r"$t$")
                        # plt.legend()
                        # plt.savefig(f"response_to_x0.svg")
                        # plt.show()

                        # biases_list = model.get_biases(past_context) #shape (1, number_bias)
                        # print(f"Biases list length: {len(biases_list)}")
                        # print(f"Biases shape: {biases_list[0].shape}")
                        # print(f"Biases: {biases_list[0]}")

                        # bias_rows = [bias.detach().cpu().squeeze(0).numpy() for bias in biases_list]
                        # max_biases = max(row.shape[0] for row in bias_rows)
                        # bias_matrix = np.full((len(bias_rows), max_biases), np.nan, dtype=np.float32)

                        # for layer_index, row in enumerate(bias_rows):
                        #     bias_matrix[layer_index, :row.shape[0]] = row

                        # masked_bias_matrix = np.ma.masked_invalid(bias_matrix)

                        # fig, ax = plt.subplots(figsize=(8,5), facecolor="white")
                        # ax.set_facecolor("white")
                        # cmap = plt.cm.viridis.copy()
                        # cmap.set_bad(color="white")
                        # vmin_val = np.nanmin(bias_matrix)
                        # vmax_val = np.nanmax(bias_matrix)
                        # image = ax.pcolormesh(
                        #     np.arange(max_biases + 1),
                        #     np.arange(len(bias_rows) + 1),
                        #     masked_bias_matrix,
                        #     cmap=cmap,
                        #     shading="flat",
                        #     vmin=vmin_val,
                        #     vmax=vmax_val,
                        # )
                        # ax.grid(False)
                        # ax.invert_yaxis()
                        # ax.set_yticks(np.arange(len(bias_rows)) + 0.5)
                        # ax.set_yticklabels([rf"$L_{{{i}}}$" for i in range(len(bias_rows))])
                        # xtick_positions = np.arange(0.5, max_biases, 100)
                        # ax.set_xticks(xtick_positions)
                        # ax.set_xticklabels([str(int(pos - 0.5)) for pos in xtick_positions])
                        # ax.set_xlabel(r"bias $[j]$")
                        # #ax.set_ylabel(r"layer $i$")
                        # #ax.set_title("Layer biases")
                        # for boundary in np.arange(1.0, len(bias_rows), 1.0):
                        #     ax.axhline(boundary, color="black", linewidth=1.0)
                        # cb = fig.colorbar(image, ax=ax, label="bias value")
                        # fig.canvas.draw()
                        # fig.savefig("biases_heatmap.png", bbox_inches="tight", dpi=300, facecolor="white")
                        # fig.savefig("biases_heatmap.svg", bbox_inches="tight", facecolor="white")
                        # plt.show()
                        # plt.close(fig)


                        #Controller part

                        #print(len(lower_bounds), flush=True)
                        lower_bounds = np.full((400,), -2.0, dtype=np.float32)
                        upper_bounds = np.full((400,), 2.0, dtype=np.float32)

                        bounds = scipy.optimize.Bounds(lower_bounds, upper_bounds, keep_feasible=True)

                    
                    
                
                        t_ref = torch.arange(0, T, h )
                        target = 0.1 * I_refs * torch.sqrt(torch.tensor(2.)) * torch.sin(2 * torch.pi * 50 * t_ref)
                        target = target / normalisation #Emulator target
                        target = target.unsqueeze(0).unsqueeze(-1).to(device) #shape (1, 400, 1)

                        #target = torch.roll(target, shifts=160, dims=1) #shape (1, 400, 1), roll to the left to put past window at the end of the sequence

                        
                        
                        for u in range(25):
                            past_context = pt_context[u].unsqueeze(0) #shape (1, n, 2, 400, 1)
                            #Optimizer f and J
                            f = Optimizer_f(model, target, past_window, past_context)
                            J_opt = Optimizer_J(model, target, past_window, past_context)

                            #Optimizer
                            method = "L-BFGS-B"
                            tol = 1e-7
                            
                            iter_count = [0]
                            def callback(xk):
                                if iter_count[0] % 20 == 0:
                                    print(f"iteration {iter_count[0]}")
                                iter_count[0] += 1

                            #x0 
                            [0.0]
                            THDs = []
                            MSEs = []
                            x0s = []
                            xstars = []
                            ystars =   []
                            yhats = []
                           
                                
                            x0 = x_0 / normalisation #shape (400,) 

                            #x0 = np.roll(x0, 160) #roll to the left to put past window at the end of the sequence
                            
                            res = minimize(f, x0, method=method, jac=J_opt, bounds=bounds, tol=tol, callback=callback)
                            #print(f"Optimization result: {res}")
                            solution = res.x #shape (400,)
                            
                            #Simulator response
                            solution = solution * normalisation

                            torch_solution = torch.tensor(solution, dtype=torch.float16).unsqueeze(0).unsqueeze(-1).to(device) #shape (1, 400, 1)


                            #plt.plot(torch_solution.squeeze(0).cpu().numpy() * 10.0, label=r"$x^{*}$")
                            #plt.show()

                            #torch_solution = torch.roll(torch_solution, shifts=-160, dims=1) #shape (1, 400, 1), roll to the left to put past window at the end of the sequence
                            output_star, final_i, final_c = Plant_torch.system_torch(torch_solution, params, params_bis, final_controller_params, final_initialisation)#shape (B, T)
                            THD_optimized = THD(output_star.squeeze(0).cpu().numpy(), sampling_period=1e-4)
                            #print(f"THD of the simulator response to optimized input: {THD_optimized}")
                            THDs.append(THD_optimized)
                            

                            output_star =  Plant_torch.Interpolate(output_star[:,200:])
                            #output_star =  Plant_torch.Interpolate(torch.roll(output_star[:,200:], shifts=280, dims=1))

                            
                            MSE_optimized = MSE(output_star.squeeze(0).cpu().numpy(), target.squeeze(0).squeeze(-1).cpu().numpy())
                            #print(f"MSE of the simulator response to optimized input: {MSE_optimized}")
                            MSEs.append(MSE_optimized)

                            print(f"{n}, {u}, THD: {THD_optimized}, MSE: {MSE_optimized}")

                            # #emulator response
                            # sol_emu = torch.tensor(solution, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) / normalisation #shape (1, 400, 1)
                            # #past_input = sol_emu[:, -past_window:, :]
                            # #sol_emu = torch.cat((past_input, sol_emu), dim=1) #shape (1, 448, 1)
                            # hat_y = model(sol_emu.to(device), past_context) #shape (1, 448, 1)
                            # #print(f"hat_y shape: {hat_y.shape}")
                            # hat_y = hat_y.squeeze(0).squeeze(-1).cpu().numpy() * normalisation #shape (448,)
                            # #print(f"hat_y shape: {hat_y.shape}")
                            # #hat_y = hat_y[past_window:] #shape (400,)
                            # #print(f"hat_y shape: {hat_y.shape}")

                            # x0s.append(x0 * normalisation * 10.0)
                            # xstars.append(solution * 10.0)
                            # ystars.append(output_star.squeeze(0).cpu().numpy() * 10.0)
                            # yhats.append(hat_y * 10.0)

                            

                            # plt.plot(solution * 10.0, label=r"$\hat x^{*}$")
                            
                            # plt.plot(target.squeeze(0).squeeze(-1).cpu().numpy() * normalisation * 10.0, label=r"$I_{ref}$")
                            # plt.plot(hat_y * 10.0, label=r"$\hat{y}^{*}$")
                            # plt.plot(output_star.squeeze(0).cpu().numpy() * 10.0, label=r"$y^{*}$")

                            # plt.plot([], [], ' ', label=r"THD = {:.2f}".format(THD_optimized * 100) + "%")
                            # plt.plot([], [], ' ', label=r"MSE = {:.2e}".format(MSE_optimized))

                            # plt.xlabel(r"$t$")
                            # plt.ylabel(r"$[A]$")
                            # plt.legend()
                            # plt.savefig(f"hypernet_response_{n}_{u}.svg")
                            # plt.show()
                            # #plt.close()
                            
                            

                            # plt.plot(solution * 10.0, label=r"$x^{*}$")
                            # plt.plot(x0 * normalisation * 10.0, label=rf"$x0^{{({v})}}$")
                            # plt.xlabel(r"$t$")
                            # plt.ylabel(r"$[A]$")
                            # plt.legend()
                            # plt.savefig(f"hypernet_input_{n}_{u}.svg")
                            # plt.show()
                            # #plt.close()



    exit()
    
                        plt.savefig(f"hypernet_input_{v}.svg")
                        plt.show()
                        #plt.close()
