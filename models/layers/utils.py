import torch
from models.layers.bnns import BayesianLinear, BayesianConv2d, BayesianBatchNorm2d

def kl_div(bnn_model):
    
    # Compute KL divergence between posterior (q) and prior (p)
    kl_divergence = 0
    for layer in bnn_model:
        if isinstance(layer, (BayesianLinear, BayesianConv2d)):
            q_dist = torch.distributions.Normal(layer.mu, torch.log1p(torch.exp(layer.rho)))  # Variational posterior
            p_dist = torch.distributions.Normal(torch.zeros_like(layer.mu), torch.ones_like(layer.mu))  # Standard normal prior
            kl_divergence += torch.distributions.kl.kl_divergence(q_dist, p_dist).sum()
            p_dist = torch.distributions.Normal(torch.zeros_like(layer.mu_bias), torch.ones_like(layer.mu_bias))  # Standard normal prior
            q_dist = torch.distributions.Normal(layer.mu_bias, torch.log1p(torch.exp(layer.rho_bias)))  # Variational posterior
            kl_divergence += torch.distributions.kl.kl_divergence(q_dist, p_dist).sum()
        
        elif isinstance(layer, BayesianBatchNorm2d):
            q_dist = torch.distributions.Normal(layer.mu_gamma, torch.log1p(torch.exp(layer.rho_gamma)))
            p_dist = torch.distributions.Normal(torch.zeros_like(layer.mu_gamma), torch.ones_like(layer.mu_gamma))
            kl_divergence += torch.distributions.kl.kl_divergence(q_dist, p_dist).sum()

            q_dist = torch.distributions.Normal(layer.mu_beta, torch.log1p(torch.exp(layer.rho_beta)))
            p_dist = torch.distributions.Normal(torch.zeros_like(layer.mu_beta), torch.ones_like(layer.rho_beta))
            kl_divergence += torch.distributions.kl.kl_divergence(q_dist, p_dist).sum()

    return kl_divergence  # Minimize ELBO
