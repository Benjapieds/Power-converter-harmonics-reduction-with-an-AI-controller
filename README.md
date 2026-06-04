# Power-converter-harmonics-reduction-with-an-AI-controller
This master’s thesis, carried out as part of an internship at CE+T Power, aims to develop an AI controller and to assess to what extent it can effectively reduce the harmonic distortion produced by a power converter connected to the main electrical grid.

see the .pdf for more information. 

The software developped is regrouped in two folders:

## KnownSystem
### Plant_torch.py
The simulator of the VSI

### controller.py
The controller (L-BFGS algorithm) using an unmodulated surrogate
### convNextBlocks.py
The U-Net architecture of the unmodulated predictor network

### learn_model.py
Example of how to train an unmodulated surrogate
### trainer_model.py
Trainer class to handle unmodulated model training

## UnknownSystem
### baseline.py
The baseline control algorithm (PI-like)
### controller.py
The controller (L-BFGS algorithm) using a modulated surrogate
### data_model.pu
Data class to handle datasets and creation of random past trajectories
### deepset_convNeXt.py
Architecture of the modulated surrogate (hyper-network + predictor)
### learn_model.py
Example of how to train a modulated surrogate
### trainer_model.py
Trainer class to handle unmodulated model training
