# A translation of https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
# from TensorFlow with some help from https://github.com/rosinality/denoising-diffusion-pytorch

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from typing import Tuple, Callable, Optional

def get_sigma_schedule(n_timestep: int, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0) -> torch.Tensor:
    """
    EDM sigma schedule following Karras et al.
    """
    t = torch.linspace(0, 1, n_timestep, dtype=torch.float64)
    sigma = sigma_min ** (1/rho) + t * (sigma_max ** (1/rho) - sigma_min ** (1/rho))
    return sigma ** rho

def extract(input: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract values from a tensor at specified timesteps."""
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def get_beta_schedule(
    schedule: str,
    start: float,
    end: float,
    n_timestep: int,
    cosine_s: float = 8e-3
) -> torch.Tensor:
    """
    Get noise schedule. EDM paper recommends using sigma-based scheduling.
    """
    if schedule == "edm":
        return get_sigma_schedule(n_timestep)
        
    def _warmup_beta(start: float, end: float, n_timestep: int, warmup_frac: float) -> torch.Tensor:
        betas = end * torch.ones(n_timestep, dtype=torch.float64)
        warmup_time = int(n_timestep * warmup_frac)
        betas[:warmup_time] = torch.linspace(start, end, warmup_time, dtype=torch.float64)
        return betas

    # Keep existing schedules for compatibility
    if schedule == "quad":
        betas = torch.linspace(start**0.5, end**0.5, n_timestep, dtype=torch.float64) ** 2
    elif schedule == "linear":
        betas = torch.linspace(start, end, n_timestep, dtype=torch.float64)
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule}")

    return betas

class Diffusion(nn.Module):
    def __init__(
        self,
        beta_type: str = "edm",
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        n_timestep: int = 40,
        model_mean_type: str = "eps",
        loss_type: str = "l2",
    ):
        """EDM-style diffusion model initialization"""
        super().__init__()
        
        # EDM uses sigma-based noise scheduling
        self.sigmas = get_sigma_schedule(
            n_timestep=n_timestep,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
        self.num_timesteps = n_timestep
        self.model_mean_type = model_mean_type
        self.loss_fn = {"l1": nn.L1Loss(), "l2": nn.MSELoss()}[loss_type]
        
        # Pre-compute EDM-specific parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.c_skip = 1
        self.c_out = -1
        self.c_in = 1
        
    def register(self, name: str, tensor: torch.Tensor):
        """Register a persistent buffer"""
        self.register_buffer(name, tensor.type(torch.float32))

    def q_sample(self, x_0: torch.Tensor, sigma: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add noise to x_0 using EDM formulation"""
        if noise is None:
            noise = torch.randn_like(x_0)
        return x_0 + noise * sigma.view(-1, 1, 1, 1)

    def get_scalings(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get EDM scaling factors"""
        c_skip = self.c_skip * torch.ones_like(sigma)
        c_out = self.c_out * torch.ones_like(sigma)
        c_in = self.c_in * torch.ones_like(sigma)
        return c_skip, c_out, c_in

    def loss(self, model: nn.Module, x_0: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """EDM training loss"""
        if sigma is None:
            sigma = torch.rand(x_0.shape[0], device=x_0.device) * (self.sigma_max - self.sigma_min) + self.sigma_min
            
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, sigma, noise)
        
        # Get model prediction
        c_skip, c_out, c_in = self.get_scalings(sigma)
        model_output = model(x_noisy, sigma)
        
        # EDM loss computation
        target = noise
        loss = self.loss_fn(model_output, target)
        
        return loss

    def predict_noise(self, model: nn.Module, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Predict noise using EDM model"""
        c_skip, c_out, c_in = self.get_scalings(sigma)
        F_x = model(x, sigma)
        return F_x

    @torch.no_grad()
    def sample_euler(
        self, 
        model: nn.Module, 
        shape: Tuple[int, ...],
        device: torch.device,
        noise_fn: Callable = torch.randn,
        num_steps: int = 40,
        **kwargs
    ) -> torch.Tensor:
        """EDM sampling using Euler solver"""
        x = noise_fn(shape, dtype=torch.float32).to(device) * self.sigma_max
        
        # Time steps follow EDM paper
        steps = torch.linspace(0, 1, num_steps + 1, device=device)
        for i in range(num_steps):
            # Get sigma for current and next step
            sigma = self.sigma_max * (1 - steps[i])
            sigma_next = self.sigma_max * (1 - steps[i + 1])
            
            # Euler step
            denoised = x - sigma * self.predict_noise(model, x, sigma.expand(shape[0]))
            d = (x - denoised) / sigma
            x = x + (sigma_next - sigma) * d
            
        return x

    @torch.no_grad()
    def sample_heun(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        noise_fn: Callable = torch.randn,
        num_steps: int = 40,
        **kwargs
    ) -> torch.Tensor:
        """EDM sampling using Heun's solver (2nd order)"""
        x = noise_fn(shape, dtype=torch.float32).to(device) * self.sigma_max
        
        # Time steps follow EDM paper
        steps = torch.linspace(0, 1, num_steps + 1, device=device)
        for i in range(num_steps):
            sigma = self.sigma_max * (1 - steps[i])
            sigma_next = self.sigma_max * (1 - steps[i + 1])
            
            # First step (Euler)
            denoised = x - sigma * self.predict_noise(model, x, sigma.expand(shape[0]))
            d = (x - denoised) / sigma
            x_euler = x + (sigma_next - sigma) * d
            
            # Second step (Heun)
            denoised_next = x_euler - sigma_next * self.predict_noise(
                model, x_euler, sigma_next.expand(shape[0])
            )
            d_next = (x_euler - denoised_next) / sigma_next
            x = x + (sigma_next - sigma) * (d + d_next) / 2
            
        return x

    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        method: str = "heun",
        **kwargs
    ) -> torch.Tensor:
        """Main sampling interface"""
        if method == "euler":
            return self.sample_euler(model, shape, device, **kwargs)
        elif method == "heun":
            return self.sample_heun(model, shape, device, **kwargs)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    @torch.no_grad()
    def sample_edm_optimized(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        noise_fn: Callable = torch.randn,
        num_steps: int = 40,
        lr: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """EDM optimized sampling with super-resolution constraint"""
        
        x = noise_fn(shape, dtype=torch.float32).to(device) * self.sigma_max
        x0_preds = []
        loss_fn_mse = nn.MSELoss()
        
        # Time steps follow EDM paper
        steps = torch.linspace(0, 1, num_steps + 1, device=device)
        for i in range(num_steps):
            sigma = self.sigma_max * (1 - steps[i])
            sigma_next = self.sigma_max * (1 - steps[i + 1])
            
            # EDM denoising step
            denoised = x - sigma * self.predict_noise(model, x, sigma.expand(shape[0]))
            d = (x - denoised) / sigma
            x_next = x + (sigma_next - sigma) * d
            
            if lr is not None:
                # Super-resolution consistency optimization
                x_next.requires_grad = True
                optimizer = optim.Adam([x_next], lr=0.05 * sigma.item()/self.sigma_max)
                
                downscaled = F.avg_pool2d(x_next, kernel_size=4, stride=4)
                loss = loss_fn_mse(downscaled.float(), lr.float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            x = x_next
            if i % 5 == 0:  # Store predictions less frequently
                x0_preds.append(denoised)
                
        return x, torch.stack(x0_preds)

    def get_loss(
        self,
        model: nn.Module, 
        x_0: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """EDM training loss"""
        
        # Sample sigma uniformly between [sigma_min, sigma_max]
        if sigma is None:
            sigma = torch.rand_like(x_0[:, 0, 0, 0]) * (self.sigma_max - self.sigma_min) + self.sigma_min
            
        # Add noise
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, sigma, noise)
        
        # Get model prediction
        noise_pred = model(x_noisy, sigma)
        
        # EDM loss computation
        return self.loss_fn(noise_pred, noise)