import torch

class BayesianLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Variational parameters (mean and log variance of weights)
        self.mu = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.mu_bias = torch.nn.Parameter(torch.randn(out_features))
        
        self.rho = torch.nn.Parameter(-5 + 0.1 * torch.randn(out_features, in_features))
        self.rho_bias = torch.nn.Parameter(-5 + 0.1 * torch.randn(out_features))
        
    def forward(self, x):
        # Compute standard deviation using softplus to ensure positivity
        sigma = torch.log1p(torch.exp(self.rho))
        sigma_bias = torch.log1p(torch.exp(self.rho_bias))
        
        # Sample weights and bias using reparameterization trick
        epsilon_w = torch.randn_like(sigma, device=sigma.device)
        epsilon_b = torch.randn_like(sigma_bias, device=sigma_bias.device)
        
        weights = self.mu + sigma * epsilon_w
        bias = self.mu_bias + sigma_bias * epsilon_b
        
        return torch.nn.functional.linear(x, weights, bias)
    

class BayesianConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        assert bias == False
        
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Variational parameters (mean and log variance of weights)
        self.mu = torch.nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.mu_bias = torch.nn.Parameter(torch.randn(out_channels))
        
        self.rho = torch.nn.Parameter(-5 + 0.1 * torch.randn(out_channels, in_channels, *self.kernel_size))
        self.rho_bias = torch.nn.Parameter(-5 + 0.1 * torch.randn(out_channels))
    
    def forward(self, x):
        # Compute standard deviation using softplus to ensure positivity
        sigma = torch.log1p(torch.exp(self.rho))
        sigma_bias = torch.log1p(torch.exp(self.rho_bias))
        
        # Sample weights and bias using reparameterization trick
        epsilon_w = torch.randn_like(sigma, device=sigma.device)
        epsilon_b = torch.randn_like(sigma_bias, device=sigma_bias.device)
        
        weights = self.mu + sigma * epsilon_w
        bias = self.mu_bias + sigma_bias * epsilon_b
        
        return torch.nn.functional.conv2d(x, weights, bias, stride=self.stride, padding=self.padding)


class BayesianBatchNorm2d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        
        self.eps = eps
        self.momentum = momentum
        
        self.mu_gamma = torch.nn.Parameter(torch.ones(num_features))
        self.mu_beta = torch.nn.Parameter(torch.zeros(num_features))
        
        self.rho_gamma = torch.nn.Parameter(-5 + 0.1 * torch.randn(num_features))
        self.rho_beta = torch.nn.Parameter(-5 + 0.1 * torch.randn(num_features))
        
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.view(1, -1, 1, 1).to(x.device)
            var = self.running_var.view(1, -1, 1, 1).to(x.device)
        
        sigma_gamma = torch.log1p(torch.exp(self.rho_gamma))
        sigma_beta = torch.log1p(torch.exp(self.rho_beta))
        
        epsilon_gamma = torch.randn_like(sigma_gamma, device=sigma_gamma.device)
        epsilon_beta = torch.randn_like(sigma_beta, device=sigma_beta.device)
        
        gamma = self.mu_gamma + sigma_gamma * epsilon_gamma
        beta = self.mu_beta + sigma_beta * epsilon_beta
        
        return torch.nn.functional.batch_norm(x, mean, var, gamma, beta, training=self.training, momentum=self.momentum, eps=self.eps)
