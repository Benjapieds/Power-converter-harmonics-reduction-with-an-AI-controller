import torch

import numpy as np
import matplotlib.pyplot as plt

from variability import Ls, Kps, Kis, I_refs, sample_Kp, sample_Ki, sample_window

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

def reference_generator(error, Kp, Ki, window):
    """
    Batched outer-loop reference generator.

    error tensor shape: (B, T)
    Kp tensor shape: (B)
    Ki tensor shape: (B)
    window tensor shape: (B) with integer values >= 0
    """
    dtype = error.dtype

    window = window.to(torch.int64)

    prop = Kp.unsqueeze(1) * error
    integral = torch.zeros_like(error)

    max_window = int(window.max().item())
    for i in range(1, max_window + 1):
        mask = (window >= i).to(dtype).unsqueeze(1) # batch elem accumulates only untill window_i
        integral += torch.roll(error, shifts=i, dims=1) * mask

    window_float = window.to(dtype).unsqueeze(1)
    integral = torch.where(window_float > 0, integral / window_float, torch.zeros_like(integral))#mean if wd > 0
    out = prop + Ki.unsqueeze(1) * integral
    return out

#interpolate the 10khz to 20khz (linear interpolation)
def Interpolate(s):
    """
    s: torch.tensor shape (B,T)
    """
    shifted = torch.roll(s, shifts=1, dims=-1)
    average = (s + shifted) / 2.
    merged = torch.roll(torch.stack((average, s), dim=-1).reshape(*s.shape[:-1], -1), shifts=-1, dims=-1)
    
    return merged

#error with reference (in: in 10kHz, out: in 20kHz)
def Error(out, I_refs, shift = True):
    """
    out : torch.tensor (B, T) with T = 400 for 20kHz sampling
    I_refs : torch.tensor (B,) the reference current amplitude for each batch element
    Shift : if need a shift of error for causality (x_t-1 => y_t)
    """
   
    # take the last period
    troncated = out
    interp = Interpolate(troncated)

    freq = 20e3 #20kHz for ref
    h = 1. / freq
    T = 20e-3
    t_ref = torch.arange(0, T, h, dtype=out.dtype, device=out.device) # shape (T)
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=out.dtype, device=out.device))
    ref_20khz = 0.1 * I_refs.unsqueeze(1) * sqrt2 * torch.sin(2 * np.pi * 50 * t_ref) #shape (B, T)

    err_20khz = ref_20khz - interp

    if shift == True:
        err_20khz = torch.roll(err_20khz, shifts=-1, dims=-1)

    return err_20khz



def controller_torch(error, params):
    """
    error : torch.tensor (B,)
    params : [Kp, Ki, saturation_up, saturation_down, past_error, integral_error]
        shape : torch.tensor (B, 6)
    return control_signal : torch.tensor (B,)
    """

    Kp = params[:, 0]
    Ki = params[:, 1]
    saturation_up = params[:, 2]
    saturation_down = params[:, 3]
    past_error = params[:, 4]
    integral_error = params[:, 5]

    # Tustin discretization (trapezoidal integration)
    integral_error = Ki * (error + past_error) / 2.0 + integral_error
    prop_error = Kp * error
    control_signal = prop_error + integral_error

    # Anti-windup 
    upper_mask = control_signal >= saturation_up
    integral_error = torch.where(upper_mask, saturation_up - prop_error, integral_error)
    control_signal = torch.where(upper_mask, saturation_up, control_signal)

    lower_mask = control_signal < saturation_down
    integral_error = torch.where(lower_mask, saturation_down - prop_error, integral_error)
    control_signal = torch.where(lower_mask, saturation_down, control_signal)

    params[:, 4] = error
    params[:, 5] = integral_error
    return control_signal
    

def ODE_torch(idx, initialization, params, params_bis):
    """
    idx : int, index of the current time step
    initialization : Io, dIo, u_reset
        shape : torch.tensor (B, 3)
    params : plant params: Rs, Ls, Es
        shape : torch.tensor (B, 3, Clock_step)
    params_bis : Vdc, bool, Transducer
        shape : torch.tensor (B, 3)
    """
    
    Io = initialization[:, 0]

    Rs = params[:, 0, idx]
    Ls = params[:, 1, idx]
    Es = params[:, 2, idx]

    Vdc = params_bis[:, 0]
    pwm_bool = params_bis[:, 1]

    dIo = (1.0 / Ls) * (-Rs * Io - Es - Vdc + 2.0 * pwm_bool * Vdc)
    return dIo

#Global static variables declaration
#Grid params
f_grid = 50.0 #Hz
T_grid = 1 / f_grid # 20ms

#PWM module params
f_clock = 1280e3 #Hz
T_clock = 1 / f_clock 
PWM_resolution = 64 #6 bits => 64 ticks for 50us
PWM_peak = 4.0

#Controller params
T_sample = 50e-6 # 20 kHz sampling frequency 
C_resolution = int(T_grid / T_sample) # 400 samples on 1 period

#OUTER controller params
COUT_T_sample = 1e-4 # 10 kHz sampling frequency for outer controller => 200 samples per 20ms period
COUT_resolution = 2 # COUT_T_sample / T_sample



#on va devoir passer le u_reset entre les périodes

#input, 1 periode complete où tout est chaneable, output, 2 periodes complete.
#System params : initial condition of current, initial dIo,  initial condition of u_reset, plant params, PWM params, inner controller params
#Cout params : sampling frequency of outer controller 
#input sequence in float 16

#manque implement COUT_params

def generate_Rs_Ls_Es_torch(R_down, R_up, t_R, d_R, Ls):
    """
    R_down shape : torch.tensor (B,)
    R_up shape : torch.tensor (B,)
    t_R shape : torch.tensor (B,)   # start time of R_up within the grid period
    d_R shape : torch.tensor (B,)   # duration of R_up (seconds), always <= T_grid for meaningful behaviour
    Ls shape : torch.tensor (B,)

    Returns: torch.tensor of shape (B, 3, Clock_step) with entries [Rs, Ls, Es]
    Behavior: default R_down everywhere; for each batch element, set Rs=R_up for a contiguous duration
    of length `d_R` starting at `t_R` (periodic over `T_grid`). If `t_R + d_R` exceeds `T_grid`, the
    interval wraps around the grid boundary but still spans exactly `d_R` seconds.
    """
    clock_steps = int(2.0 * T_grid / T_clock)
    t = torch.arange(clock_steps, dtype=torch.float32, device=t_R.device) * T_clock  # shape : (Clock_step,)

    # Fold time into one grid period.
    t_periodic = torch.remainder(t, T_grid)  # shape: (Clock_step,)

    # Broadcasted interval boundaries
    t_start = t_R.unsqueeze(1)  # shape: (B,1)
    duration = d_R.unsqueeze(1)  # shape: (B,1)
    t_end = t_start + duration    # shape: (B,1)

    # Handle per-batch interval selection:
    # - no wrap (t_end <= T_grid) -> select t in [t_start, t_end)
    # - wrap (t_end > T_grid) -> select t >= t_start OR t < (t_end - T_grid)
    end_within = (t_end <= T_grid)

    # Compute masks (t_periodic will broadcast against (B,1) shapes)
    mask_no_wrap = (t_periodic >= t_start) & (t_periodic < t_end)
    mask_wrap = (t_periodic >= t_start) | (t_periodic < (t_end - T_grid))

    # Combine masks per-batch
    mask = torch.where(end_within, mask_no_wrap, mask_wrap)

    Rs = torch.where(mask, R_up.unsqueeze(1), R_down.unsqueeze(1)) # shape : (B, Clock_step)
    Ls_tensor = Ls.unsqueeze(1).repeat(1, clock_steps) # shape : (B, Clock_step)

    Es = 120. * np.sqrt(2) * torch.sin(2 * np.pi * 50 * t) + 12. * np.sqrt(2) * torch.sin(2 * np.pi * 150 * t) \
        + 8. * np.sqrt(2) * torch.sin(2 * np.pi * 250 * t) + 2. * np.sqrt(2) * torch.sin(2 * np.pi * 350 * t)
    #shape : (Clock_step,)

    Es = Es.unsqueeze(0).repeat(R_down.size(0), 1) # shape : (B, Clock_step)
    out = torch.stack((Rs, Ls_tensor, Es), dim=1) # shape : (B, 3, Clock_step)
    return out

def generate_Vdc_bool_Transducer_torch(B, device=None):
    """
    B : int, batch size
    out: shape : torch.tensor (B, 3) for Vdc, bool, Transducer
    """
    Vdc = torch.full((B,), 250.0, dtype=torch.float32, device=device)
    bool = torch.ones((B,), dtype=torch.float32, device=device)#start with 1
    Transducer = torch.full((B,), 0.1, dtype=torch.float32, device=device)# 0.1 as G
    out = torch.stack((Vdc, bool, Transducer), dim=1) # shape : (B, 3)
    return out

def generate_controller_params_torch(Kp, Ki):
    """
    Kp shape : torch.tensor (B,)
    Ki shape : torch.tensor (B,)
    out: shape : torch.tensor (B, 6) for Kp, Ki, saturation_up, saturation_down, past_error, integral_error
    """
    B = Kp.size(0)
    device = Kp.device
    dtype = torch.float32
    past_error = torch.zeros((B,), dtype=dtype, device=device)# init to zero
    integral_error = torch.zeros((B,), dtype=dtype, device=device)
    saturation_up = torch.zeros((B,), dtype=dtype, device=device) #will be updated at each sampling step
    saturation_down = torch.zeros((B,), dtype=dtype, device=device) #will be updated at each sampling step
    out = torch.stack((Kp, Ki, saturation_up, saturation_down, past_error, integral_error), dim=1) # shape : (B, 6)
    return out

def generate_initialization_torch(B, device=None):
    """
    Io shape : torch.tensor (B,)
    dIo shape : torch.tensor (B,)
    u_reset shape : torch.tensor (B,)

    out: shape : torch.tensor (B, 3) for Io, dIo, u_reset
    """
    Io = torch.zeros((B,), dtype=torch.float32, device=device)
    dIo = torch.zeros((B,), dtype=torch.float32, device=device)
    u_reset = torch.full((B,), float(PWM_resolution), dtype=torch.float32, device=device)#  start with reset at MAX
    out = torch.stack((Io, dIo, u_reset), dim=1) # shape : (B, 3)
    return out

def generate_input_sequence_torch(I_refs):
    """
    I_refs shape : torch.tensor (B,)

    out: shape : torch.tensor (B, C_resolution) for 1 period of reference signal for the inner controller, sampled at T_sample (20kHz)
    """
    t_ref = torch.arange(0, T_grid, T_sample, dtype=I_refs.dtype, device=I_refs.device) # shape : (C_resolution,)
    ref = 0.1 * I_refs.unsqueeze(1) * np.sqrt(2.) * torch.sin(2 * np.pi * 50 * t_ref) # shape : (B, C_resolution)
    return ref

def system_torch(input_sequence, params, params_bis, controller_params, initialization):
    """
    input_sequence : 1 period of reference signal for the inner controller, sampled at T_sample (20kHz)
        shape : torch.tensor (B, C_resolution)
    params : plant params: Rs, Ls, Es
        shape : torch.tensor (B, 3, Clock_step)  
    params_bis : Vdc, bool, Transducer
        shape : torch.tensor (B, 3)
    controller_params : Kp, Ki, saturation_up, saturation_down, past_error, integral_error
        shape : torch.tensor (B, 6)
    initialization : Io, dIo, u_reset
        shape : torch.tensor (B, 3)
    """

    output_sequence = torch.zeros(
        input_sequence.size(0),
        int(2.0 / COUT_resolution) * input_sequence.size(1) // 2,
        dtype=torch.float16,
        device=input_sequence.device,
    ) #shape : (B, 2 * C_resolution) for 2 periods of output signal for the inner controller, sampled at T_sample (20kHz)
    
    output_index = 0

    u = 0 #PWM counter
    COUT_u = 2 #Outer controller counter
    C_ref = 0 #Inner controler reference counter

    clock_steps = int(2.0 * T_grid / T_clock) // 2
    
    #Simulate for 2 grid periods, resolution of T_clock
    for idx in range(clock_steps):
        
        #PWM module down mechansim
        params_bis[:, 1] = torch.where(initialization[:, 2] > u, 1.0, 0.0)
        #print(f"Time step {idx}, u: {u}", flush=True)
        #print(f"Time step {idx}, PWM bool: {params_bis[:, 1]}", flush=True)
        #print(f"Time step {idx}, PWM signal: {initialization[:, 2]}", flush=True)
        #Controler new sampling period, 20kHz tick
        if u % PWM_resolution == 0 :

            u = 0 #reset internal clock
            params_bis[:, 1] = torch.ones(params_bis.shape[0], dtype=params_bis.dtype, device=params_bis.device)

            sample = torch.mul(initialization[:, 0], params_bis[:, 2]).to(torch.float16)
            #In 16 bits as 12 bits does bot exists

            #reference definition
            reference = input_sequence[:, C_ref]
            C_ref += 1
            if C_ref == C_resolution : C_ref = 0 #Reset for second period

            
            error = reference - sample

            controller_params[:, 2] = PWM_peak / PWM_resolution * (PWM_resolution - PWM_resolution // 2.) - reference
            controller_params[:, 3] = PWM_peak / PWM_resolution * (0. - PWM_resolution // 2.) - reference

            control_signal = controller_torch(error, controller_params)
            
            #reconstruction PWM signal, set of PWM modulation
            initialization[:, 2] = (control_signal + reference) / PWM_peak * PWM_resolution + PWM_resolution // 2.  #take into account cpeak and resolution

            # Neglect computational delay: disable PWM where reset is zero.
            params_bis[:, 1] = torch.where(
                initialization[:, 2] == 0.0,
                torch.zeros_like(params_bis[:, 1]),
                params_bis[:, 1],
            )
            
            #Second controller sampling
            if COUT_u % COUT_resolution == 0: #To go from 20khz to 10khz 
                COUT_u = 1
                output_sequence[:,output_index] = sample
                output_index += 1
            else :
                COUT_u +=1
    
        #Euler
        initialization[:, 1] = ODE_torch(idx, initialization, params, params_bis)
        initialization[:, 0] = initialization[:, 0] + T_clock * initialization[:, 1]
        
        #Next clock tick       
        u += 1
    return output_sequence, initialization, controller_params

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    
    Rs_up = 7.0
    Rs_down = 1.0
    t_Rs = 0.0158362
    ds = 0.0103472
    print(f"Rs_up: {Rs_up}, Rs_down: {Rs_down}, t_Rs: {t_Rs}, d: {ds}")



    batch_size = 2
   
    Iref = torch.as_tensor(I_refs, dtype=torch.float32).flatten().to(device).repeat(batch_size) # shape (10000)
    Kp = torch.tensor([Kps], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    Ki = torch.tensor([Kis], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    R_down = torch.tensor([Rs_down], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    R_up = torch.tensor([Rs_up], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    t_R = torch.tensor([t_Rs], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    d_R = torch.tensor([ds], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)
    L = torch.tensor([Ls], dtype=torch.float32).to(device).repeat(batch_size) # shape (10000,)

    input_sequence = generate_input_sequence_torch(Iref).to(device) #shape (10000)
    initialisation = generate_initialization_torch(batch_size).to(device)
    print(f"input_sequence shape: {input_sequence.shape}")
    print(f"Initialisation shape: {initialisation.shape}")
    controller_params = generate_controller_params_torch(Kp, Ki)
    params_bis = generate_Vdc_bool_Transducer_torch(batch_size).to(device)
    print(f"Params bis shape: {params_bis.shape}")
    params =  generate_Rs_Ls_Es_torch(R_down, R_up, t_R, d_R, L)
    print(f"Params shape: {params.shape}")



    response_system = []

    #Generate with default reference

    #good
    # Cp = 0.35
    # Ci = 0.15
    # Ki = 0.00
    # phi = 3

    #not good
    # Cp = 1.20
    # Ci = 0.15
    # Ki = 0.0
    # phi = 3

    # #impossible
    Cp = 1.20
    Ci = 0.15
    Ki = 0.0
    phi = 3 #With torch.roll

    thd_arrays = []
    input_arrays = []
    error_buffer = torch.zeros_like(input_sequence)
    t_1 = range(0,400,1)
    t_2 = range(0,400,2)
    #Disguard simulation transient
    first_input_sequence = input_sequence.clone()
    input_arrays.append(input_sequence[0].cpu())
    output_sequence, initialisation, controller_params = system_torch(input_sequence, params, params_bis, controller_params, initialisation)#shape (B, T)
    for j in range(11):

        for _ in range(2):
            output_sequence, initialisation, controller_params = system_torch(input_sequence, params, params_bis, controller_params, initialisation)#shape (B, T)
        error = Error(output_sequence, Iref, shift=True)#shape (B, T)
        #plt.plot(t_1, input_sequence[0].cpu().numpy(), label="Reference")
        for u in range(len(error_buffer)):
            error_buffer[:,u] += error[:,u]

        error = torch.roll(error, shifts=+1, dims=-1) #shift for causality => not needed as boundary condition on System is 0 = 400t previous
        max_window = int(phi)
        integral = torch.zeros_like(error)
        for i in range(1, max_window + 1):
            shifted = torch.cat(
                [torch.zeros_like(error[:, :i]), error[:, :-i]],
                dim=1,
            )
            integral += shifted
        integral = integral / max_window

        
        
        #input_sequence = first_input_sequence + Cp * error + Ci * error_buffer + Ki * integral
        input_sequence = input_sequence + Cp * error + Ki * integral
        np_out = output_sequence.cpu().numpy()
        np_out = np.concatenate((np_out, np_out), axis=1)
        THD_value = THD(np_out[0], 1e-4)
        print(f"THD of the output signal: {THD_value}")
        thd_arrays.append(THD_value)

        response_system.append(output_sequence[0].cpu())
        #Store response to sinus and metadata
        #troncated = output_sequence[:, 200:]
        #interp = Interpolate(troncated)

      
        
        # plt.plot(t_2,output_sequence[0].cpu().numpy(), label="Output")
        # plt.plot(t_1, error[0].cpu().numpy(), label="Error")
        # plt.plot(input_sequence[0].cpu().numpy(), label="Input sequence next")
        # plt.legend()
       
        # plt.show()
        input_arrays.append(input_sequence[0].cpu())
        

        #input_sequence = input_sequence + error
        
    flatten_response = torch.cat(response_system, dim=0)
    flatten_input = torch.cat(input_arrays, dim=0)
    t_in = range(0, 400*3, 1)
    t_out = range(0, 400*3, 2)
    from matplotlib_init import init_matplotlib
    init_matplotlib()
    plt.plot(t_in,flatten_input.cpu().numpy()[:400*3] * 10.0, label=r"$x^{0:2}$")
    plt.plot(t_out,flatten_response.cpu().numpy()[:200*3] * 10.0, label=r"$y^{0:2}$")
    plt.vlines(x=400, ymin=-9., ymax=9.0, colors='gray', linestyles='dashed', linewidth=0.5)
    plt.vlines(x=800, ymin=-9., ymax=9.0, colors='gray', linestyles='dashed', linewidth=0.5)

    plt.yticks([-5, 0, 5])
    plt.ylim(-8.5, 8.5)
    plt.xticks(np.arange(0, 1201, 200))
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$[A]$")
    plt.savefig("Baseline.svg")
    plt.show()

    #The 5th respone
    t_1 = range(0,400,1)
    t_2 = range(0,400,2)
    plt.plot(t_1, flatten_input.cpu().numpy()[2000:2400] * 10.0, label=r"$x^{5}$")
    plt.plot(t_2, flatten_response.cpu().numpy()[1000:1200] * 10.0, label=r"$y^{5}$")
    plt.yticks([-5, 0, 5])
    plt.ylim(-8.5, 8.5)
    plt.xticks(np.arange(0, 401, 50))
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$[A]$")
    plt.savefig("Baseline5.svg")
    plt.show()

    #The 10th respone
    plt.plot(t_1, flatten_input.cpu().numpy()[4000:4400] * 10.0, label=r"$x^{10}$")
    plt.plot(t_2, flatten_response.cpu().numpy()[2000:2200] * 10.0, label=r"$y^{10}$")
    plt.yticks([-5, 0, 5])
    plt.ylim(-8.5, 8.5)
    plt.xticks(np.arange(0, 401, 50))
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$[A]$")
    plt.savefig("Baseline10.svg")
    plt.show()
    
