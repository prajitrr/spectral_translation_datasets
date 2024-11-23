import random
import imageio
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from helpers import *



def training_loop(ddpm, 
                  loader, 
                  n_epochs, 
                  optim, device, 
                  display=False,
                  store_path="ddpm_model.pt", 
                  matrix_dim = 28,
                  bond_tolerance = 1/1.1,
                  beta_timestep=1, 
                  beta_atom=1,
                  beta_bond=1):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps
    diag_ind = np.diag_indices(matrix_dim)
    zero_tensor = torch.zeros(matrix_dim, matrix_dim).to(device)

    for epoch in tqdm(range(n_epochs), 
                      desc=f"Training progress", 
                      colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, 
                                          leave=False, 
                                          desc=f"Epoch {epoch + 1}/{n_epochs}", 
                                          colour="#005500")):
            # Loading data
            #print(batch["image"].shape)
            x0 = batch["image"].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            # Additionally, the model has constraings to nudge atom numbers to integers and keep valences less than or equal to 4
            x0_zeros = 1 / x0
            x0_zeros = torch.where(x0_zeros < bond_tolerance, 
                                    0, 
                                    x0_zeros)
            x0_zeros[:, 0, diag_ind[0], diag_ind[1]] = zero_tensor
            loss = beta_timestep * mse(eta_theta, eta) \
                   + beta_atom * mse(6/torch.diagonal(x0, 0, dim1 = 2, dim2=3), 
                                     torch.round(6/torch.diagonal(x0, 0, dim1= 2, dim2=3))) \
                   + 0.5 * beta_bond * torch.mean(nn.ReLU(torch.sum(x0_zeros, 2) - 4*torch.ones(torch.sum(x0_zeros, 2).shape).to(device))) \
                   + 0.5 * beta_bond * torch.mean(nn.ReLU(torch.sum(x0_zeros, 3) - 4*torch.ones(torch.sum(x0_zeros, 3).shape).to(device))) 
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)