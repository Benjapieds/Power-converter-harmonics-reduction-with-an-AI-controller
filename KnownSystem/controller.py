import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.optimize import minimize


from Unet_blocks import UNet
from data_model import PlantData

import Plant as Plant
import Plant_torch

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

def Optimizer_f(model, target, past_window):
    """
    Return f(x) -> float with f(x) = lambda x: model.predict_loss(x, target, past_window)

    target: torch tensor of shape (1, 400, 1)
    past_window: int, the number of past time steps used as input to the model

    Take care of good shape and types for torch and numpy compatibility.
    
    """
    def f(x):
        #x is a numpy array of shape (400)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) #shape (1, 400, 1)

        loss = model.predict_loss(x_tensor, target, past_window) #shape (1,)
        return loss.item() #return a scalar

    return f

def Optimizer_J(model, target, past_window):
    """
    Return J(x) -> numpy array with J(x) = lambda x: model.get_Jacobian()(x, target, past_window)

    target: torch tensor of shape (1, 400, 1)
    past_window: int, the number of past time steps used as input to the model

    Take care of good shape and types for torch and numpy compatibility.
    
    """
    jacobian_gn = model.get_Jacobian()

    def J(x):
        # x is a numpy array of shape (400)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) #shape (1, 400, 1)
        jacobian = jacobian_gn(x, target, past_window) #shape (1, 400, 1)
        return jacobian.squeeze(0).squeeze(-1).numpy() #shape (400,)

    return J
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    from variability import I_refs, Kps, Kis, Rs_down, Rs_up, t_Rs, Ls
    from matplotlib_init import init_matplotlib
    init_matplotlib()

    batch_size = 1

    #Simuilate response to sinus
    Iref = torch.as_tensor(I_refs, dtype=torch.float32).flatten().to(device).repeat(batch_size)
    Kp = torch.tensor([Kps], dtype=torch.float32).to(device).repeat(batch_size)
    Ki = torch.tensor([Kis], dtype=torch.float32).to(device).repeat(batch_size) 
    R_down = torch.tensor([Rs_down], dtype=torch.float32).to(device).repeat(batch_size)
    R_up = torch.tensor([Rs_up], dtype=torch.float32).to(device).repeat(batch_size)
    t_R = torch.tensor([t_Rs], dtype=torch.float32).to(device).repeat(batch_size) 
    L = torch.tensor([Ls], dtype=torch.float32).to(device).repeat(batch_size) 
    

    input_sequence = Plant_torch.generate_input_sequence_torch(Iref).to(device)
    initialisation = Plant_torch.generate_initialization_torch(batch_size).to(device)
    print(f"input_sequence shape: {input_sequence.shape}")
    print(f"Initialisation shape: {initialisation.shape}")
    controller_params = Plant_torch.generate_controller_params_torch(Kp, Ki)
    params_bis = Plant_torch.generate_Vdc_bool_Transducer_torch(batch_size).to(device)
    print(f"Params bis shape: {params_bis.shape}")
    params =  Plant_torch.generate_Rs_Ls_Es_torch(R_down, R_up, t_R, L)
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
    plt.show(block=False)
    plt.close()
   


    #Performance of x0
    kps = [0.0, 0.5, 1.0, 3.0]

    #Define y_target
    normalisation = 0.9740
    past_window = 48
    freq = 20e3 #20kHz
    h = 1. / freq
    T = 20e-3 #20ms period
            
    t_ref = torch.arange(0, T, h )
    target = 0.1 * I_refs * torch.sqrt(torch.tensor(2.)) * torch.sin(2 * torch.pi * 50 * t_ref)
    target = target / normalisation #Emulator target
    target = target.unsqueeze(0).unsqueeze(-1) #shape (1, 400, 1)



    for l, k in enumerate(kps):
            
             batch_kp = torch.tensor([k], dtype=torch.float32).to(device).repeat(batch_size)
             batch_ki = torch.tensor([0.0], dtype=torch.float32).to(device).repeat(batch_size) 
             batch_window = torch.tensor([0], dtype=torch.float32).to(device).repeat(batch_size)
             input_sequence_bis = input_sequence + Plant_torch.reference_generator(error, batch_kp, batch_ki, batch_window)
             print(f"input_sequence_bis shape: {input_sequence_bis.shape}")
             x0 = input_sequence_bis #shape (400,) 
        
             output_star, final_i, final_c = Plant_torch.system_torch(x0, params, params_bis, final_controller_params, final_initialisation)#shape (B, T)
             THD_optimized = THD(output_star.squeeze(0).cpu().numpy(), sampling_period=1e-4)
             print(f"THD of the simulator response to optimized input: {THD_optimized}")
            

             output_star =  Plant_torch.Interpolate(output_star[:,200:])
             #output_star =  Plant_torch.Interpolate(torch.roll(output_star[:,200:], shifts=280, dims=1))

            
             MSE_optimized = MSE(output_star.squeeze(0).cpu().numpy(), target.squeeze(0).squeeze(-1).cpu().numpy() * normalisation)
             print(f"MSE of the simulator response to optimized input: {MSE_optimized}")
            
            

             plt.plot(x0.squeeze(0).cpu().numpy() * 10.0, label=rf"$x0^{{({l})}}$")
             plt.plot(target.squeeze(0).squeeze(-1).cpu().numpy() * normalisation * 10.0, label=r"$I_{ref}$")
             plt.plot(output_star.squeeze(0).cpu().numpy() * 10.0, label=rf"$y^{{({l})}}$")

             plt.plot([], [], ' ', label=r"THD = {:.2f}".format(THD_optimized * 100) + "%")
             plt.plot([], [], ' ', label=r"MSE = {:.2e}".format(MSE_optimized))

             plt.xlabel(r"$t$")
             plt.ylabel(r"$[A]$")
             plt.legend()
             plt.savefig(f"responsex0_{l}.svg")
             plt.show(block=False)
             plt.close()
     print("finish", flush=True)
    Load model
    TTTHHD = []
    from convNextBlocks import ConvNeXtUnet
    

    for depth in [1,2,3,4]:
         for wide in [8, 16, 32]:
             for patch_len in [3, 5, 7, 15]:
    
    
                model_root = f"COMPLETEHERE"
                model = ConvNeXtUnet(
                    lr=0.001,
                    depth=depth,
                    channels=wide,
                    patch_len=patch_len,
                    patch_stride=1,
                    patch_padding=patch_len//2,
                )

                
                Load_model(model, model_root)
                model.eval()
                print(f"Model {depth}_{wide}_{patch_len} has {model.count_parameters()}", flush =True)
                print("#########################", flush=True)



                with torch.no_grad():
                    #Boudns of optimization problem
                    root = "c:/Users/keutg/OneDrive - Universite de Liege/M2/TFE/d2l-en/pytorch/chapter_attention-mechanisms-and-transformers/dataset_bis/dataset_clean.pt"
                    expand_root = "c:/Users/keutg/OneDrive - Universite de Liege/M2/TFE/Codes/Model3/dataset_expand/expand.pt"
                    dataset = PlantData(batch_size=256, pastwindow = 48, root=root, expand_root=expand_root)

                    #bounds = dataset.get_bounds()
                    #print(bounds)
                    #lower_bounds, upper_bounds = dataset.get_bounds()

                    #print(len(lower_bounds), flush=True)
                    lower_bounds = np.full((400,), -2.0, dtype=np.float32)
                    upper_bounds = np.full((400,), 2.0, dtype=np.float32)

                    bounds = scipy.optimize.Bounds(lower_bounds, upper_bounds, keep_feasible=True)

                    #Define y_target
                    normalisation = 0.9740
                    past_window = 48
                    freq = 20e3 #20kHz
                    h = 1. / freq
                    T = 20e-3 #20ms period
            
                    t_ref = torch.arange(0, T, h )
                    target = 0.1 * I_refs * torch.sqrt(torch.tensor(2.)) * torch.sin(2 * torch.pi * 50 * t_ref)
                    target = target / normalisation #Emulator target
                    target = target.unsqueeze(0).unsqueeze(-1) #shape (1, 400, 1)

                    #target = torch.roll(target, shifts=160, dims=1) #shape (1, 400, 1), roll to the left to put past window at the end of the sequence

                    
                    

                    #Optimizer f and J
                    f = Optimizer_f(model, target, past_window)
                    J_opt = Optimizer_J(model, target, past_window)


                    #emulator response
                    sol_emu = target #shape (1, 400, 1)
                    #past_input = sol_emu[:, -past_window:, :]
                    #sol_emu = torch.cat((past_input, sol_emu), dim=1) #shape (1, 448, 1)
                    hat_y = model(sol_emu) #shape (1, 448, 1)
                    #print(f"hat_y shape: {hat_y.shape}")
                    hat_y = hat_y.squeeze(0).squeeze(-1).cpu().numpy() * normalisation #shape (448,)
                    #print(f"hat_y shape: {hat_y.shape}")
                    #hat_y = hat_y[past_window:] #shape (400,)
                    #print(f"hat_y shape: {hat_y.shape}")

                    plt.plot(target.squeeze(0).squeeze(-1).cpu().numpy() * normalisation * 10.0, label=r"$x^{sin}$")
                    plt.plot(hat_y * 10.0, label=r"$\hat{y}^{sin}$")
                    plt.ylabel(r"$[A]$")
                    plt.xlabel(r"$t$")
                    plt.legend()
                    plt.savefig(f"{depth}_{wide}_{patch_len}_response_to_sinus_model.svg")
                    plt.show(block=False)
                    plt.close()
                    
                    

                    

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
                    for v, kp in enumerate(kps):
                        batch_kp = torch.tensor([kp], dtype=torch.float32).to(device).repeat(batch_size)
                        batch_ki = torch.tensor([0.0], dtype=torch.float32).to(device).repeat(batch_size) 
                        batch_window = torch.tensor([0], dtype=torch.float32).to(device).repeat(batch_size)
                        input_sequence_bis = input_sequence + Plant_torch.reference_generator(error, batch_kp, batch_ki, batch_window)
                        #print(f"input_sequence_bis shape: {input_sequence_bis.shape}") 
                        x0 = input_sequence_bis.squeeze(0).detach().cpu().numpy() / normalisation #shape (400,)

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
                        TTTHHD.append(THD_optimized)
                        print(TTTHHD)

                        output_star =  Plant_torch.Interpolate(output_star[:,200:])
                        #output_star =  Plant_torch.Interpolate(torch.roll(output_star[:,200:], shifts=280, dims=1))

                        
                        MSE_optimized = MSE(output_star.squeeze(0).cpu().numpy(), target.squeeze(0).squeeze(-1).cpu().numpy())
                        #print(f"MSE of the simulator response to optimized input: {MSE_optimized}")
                        MSEs.append(MSE_optimized)

                        #emulator response
                        sol_emu = torch.tensor(solution, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) / normalisation #shape (1, 400, 1)
                        #past_input = sol_emu[:, -past_window:, :]
                        #sol_emu = torch.cat((past_input, sol_emu), dim=1) #shape (1, 448, 1)
                        hat_y = model(sol_emu) #shape (1, 448, 1)
                        #print(f"hat_y shape: {hat_y.shape}")
                        hat_y = hat_y.squeeze(0).squeeze(-1).cpu().numpy() * normalisation #shape (448,)
                        #print(f"hat_y shape: {hat_y.shape}")
                        #hat_y = hat_y[past_window:] #shape (400,)
                        #print(f"hat_y shape: {hat_y.shape}")

                        x0s.append(x0 * normalisation * 10.0)
                        xstars.append(solution * 10.0)
                        ystars.append(output_star.squeeze(0).cpu().numpy() * 10.0)
                        yhats.append(hat_y * 10.0)

                        

                        plt.plot(solution * 10.0, label=r"$\hat x^{*}$")
                        
                        plt.plot(target.squeeze(0).squeeze(-1).cpu().numpy() * normalisation * 10.0, label=r"$I_{ref}$")
                        plt.plot(hat_y * 10.0, label=r"$\hat{y}^{*}$")
                        plt.plot(output_star.squeeze(0).cpu().numpy() * 10.0, label=r"$y^{*}$")

                        plt.plot([], [], ' ', label=r"THD = {:.2f}".format(THD_optimized * 100) + "%")
                        plt.plot([], [], ' ', label=r"MSE = {:.2e}".format(MSE_optimized))

                        plt.xlabel(r"$t$")
                        plt.ylabel(r"$[A]$")
                        plt.legend()
                        plt.savefig(f"{depth}_{wide}_{patch_len}_response_{v}.svg")
                        plt.show(block=False)
                        plt.close()
                        
                        

                        plt.plot(solution * 10.0, label=r"$x^{*}$")
                        plt.plot(x0 * normalisation * 10.0, label=rf"$x0^{{({v})}}$")
                        plt.xlabel(r"$t$")
                        plt.ylabel(r"$[A]$")
                        plt.legend()
                        plt.savefig(f"{depth}_{wide}_{patch_len}_input_{v}.svg")
                        plt.show(block=False)
                        plt.close()
                        
                        


                
                    #print(f"THDs for kps {kps}: {THDs}")
                    #print(f"MSEs for kps {kps}: {MSEs}")
    torch.save(TTTHHD, "TTTHHD.pt")


    
