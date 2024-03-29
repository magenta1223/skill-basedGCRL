import torch
import torch.distributions as torch_dist
from ..contrib import TanhNormal

# --------------------------- Distribution --------------------------- #

def get_dist(model_output, log_scale = None, scale = None,  detached = False, tanh = False):
    if detached:
        model_output = model_output.clone().detach()
        model_output.requires_grad = False

    if log_scale is None and scale is None:
        mu, log_scale = model_output.chunk(2, -1)
        scale = log_scale.clamp(-10, 2).exp()
    else:
        mu = model_output
        if log_scale is not None:
            scale = log_scale.clamp(-10, 2).exp()

    if tanh:
        return TanhNormal(mu, scale)
    else:
        dist = torch_dist.Normal(mu, scale)
        return torch_dist.Independent(dist, 1)

def get_fixed_dist(model_output,  tanh = False):
    model_output = model_output.clone().detach()
    mu, log_scale = torch.zeros_like(model_output).chunk(2, -1)
    scale = log_scale.exp()
    if tanh:
        return TanhNormal(mu, scale)
    else:
        dist = torch_dist.Normal(mu, scale)
        return torch_dist.Independent(dist, 1)  

def get_scales(dist):
    
    assert isinstance(dist, TanhNormal) or isinstance(dist, torch_dist.Independent), "Invalid type of distributions"

    if isinstance(dist, TanhNormal):
        scale = dist._normal.base_dist.scale.abs().mean().item() 
    else:
        scale = dist.base_dist.scale.abs().mean().item() 

    return scale


def nll_dist(z, q_hat_dist, pre_tanh_value = None, tanh = False):
    if tanh:
        return - q_hat_dist.log_prob(z, pre_tanh_value)
    else:
        return - q_hat_dist.log_prob(z)

def kl_divergence(dist1, dist2, *args, **kwargs):
    return torch_dist.kl_divergence(dist1, dist2)

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)

def kl_annealing(epoch, start, end, rate=0.9):
    return end + (start - end)*(rate)**epoch

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) 
    y = y.unsqueeze(0) 
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) 

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y) 
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def to_skill_embedding(tensor):
    """
    """
    return tensor.clone().detach().cpu().squeeze(0).numpy()


def weighted_mse(pred, target, weights = None):
    agg_dims = list(range(1, len(pred.shape)))
    mse = torch.pow(pred - target, 2).mean(dim = agg_dims)
    
    if weights is not None:
        weighted_error = mse * weights
        weighted_mse_loss = torch.mean(weighted_error)
        return weighted_mse_loss
    else:
        return torch.mean(mse)
