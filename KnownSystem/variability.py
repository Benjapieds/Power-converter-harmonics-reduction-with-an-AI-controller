import numpy as np

#1. of mean 
R_down_min = 0.8
R_down_max = 1.2

#6. of mean
R_up_min = 4.8
R_up_max = 7.2

#5e-3 of mean, time for Rstep
t_min = 0.0
t_max = 10e-3

#1.2e-3 of mean
L_min = 1.e-3
L_max = 1.4e-3

""" Params inner PI """
#2.01 mean
Kp_min = 1.608
Kp_max = 2.412

#0.108 mean
Ki_min = 0.0864
Ki_max = 0.1296

#8 RMS mean
I_ref_min = 6.4
I_ref_max = 9.6

""" Params outer sampler """
#0 mean
#sample_Kp_min = 0.5

sample_Kp_min = -0.5 #=> 0.125 resolution for 25 steps
sample_Kp_max = 1.5 #=> 0.083 resolution for 25 steps

# mean
#sample_Ki_min = 0.02 # => 0.0192 resolution for 25 steps

sample_Ki_min = -0.5 # => 0.04 resolution for 25 steps
sample_Ki_max = 0.5 

#Poisso sampler dans une Poisson ou uniorme pour dataset? => qu'il apprenne fort quoi ?
sample_window_min = 0
sample_window_max = 25
#sample_window_max = 3 

size = 5 #78125 tasks
sample_size = 25 #15625 samplings per task

#sample_size = 4

R_down = np.linspace(R_down_min, R_down_max, size)
R_up = np.linspace(R_up_min, R_up_max, size)
t_R = np.linspace(t_min, t_max, size)
L = np.linspace(L_min, L_max, size)
Kp = np.linspace(Kp_min, Kp_max, size)
Ki = np.linspace(Ki_min, Ki_max, size)
I_ref = np.linspace(I_ref_min, I_ref_max, size)

sample_Kp = np.linspace(sample_Kp_min, sample_Kp_max, sample_size)
sample_Ki = np.linspace(sample_Ki_min, sample_Ki_max, sample_size)
sample_window = np.arange(sample_window_min, sample_window_max + 1, dtype=np.int16)

rg = np.random.default_rng()#seed = 1234567899
indexes = rg.integers(low=0, high=size, size=7)
sample_indexes = rg.integers(low=0, high=sample_size, size=3)

Rs_down = R_down[indexes[0]]
Rs_up = R_up[indexes[1]]
t_Rs = t_R[indexes[2]]
Ls = L[indexes[3]]
Kps = Kp[indexes[4]]
Kis = Ki[indexes[5]]
I_refs = I_ref[indexes[6]]
sample_Kps = sample_Kp[sample_indexes[0]]
sample_Kis = sample_Ki[sample_indexes[1]]
sample_windows = sample_window[sample_indexes[2]]

t_Rs = 5e-3


Rs_down = 1.0
Rs_up = 7.0
Ls = 1.2e-3
Kps = 2.01
Kis = 0.108
I_refs = 5.0
sample_Kps = 0.8
sample_Kis = 0.05
sample_windows = 20

