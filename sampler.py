import torch
import torch.distributions as dists
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_posterior_o(y, log_tau, model):  
  y.requires_grad_()
  logp = -torch.sum(model.forward(y, None), dim=-1, keepdims=True) * 10000
  grad_y = torch.autograd.grad(logp.sum(), y)[0].detach()
  delta = torch.arange(256, dtype=torch.float32).to(y.device) / 256.0 - torch.unsqueeze(y, -1)
  grad_y = torch.unsqueeze(grad_y, -1)
  rate_y = delta * grad_y - delta ** 2 / torch.exp(log_tau) / 2
  log_posterior_y = rate_y / 2.0
  return logp, log_posterior_y


def o_mcmc_step(y, log_tau, model):
  logp_current, log_posterior_y = get_posterior_o(y, log_tau, model)
  dist_y = dists.Categorical(logits=log_posterior_y)
  v_cat = dist_y.sample()
  v = v_cat.float() / 256.0
  logp_next, log_posterior_v = get_posterior_o(v, log_tau, model)
  with torch.no_grad():
    dist_v = dists.Categorical(logits=log_posterior_v)
    y_cat = (y * 256.0).to(torch.int64)
    log_forward = torch.sum(dist_y.log_prob(v_cat), (1, 2, 3)) + logp_current.view(-1)
    log_backward = torch.sum(dist_v.log_prob(y_cat), (1, 2, 3)) + logp_next.view(-1)
    log_acc = log_backward - log_forward
    accepted = (torch.rand_like(log_acc) < log_acc.exp()).float().view(-1, 1, 1, 1)
    accs = torch.clamp(log_acc, max=0).exp().mean().item()
    new_y = (1.0 - accepted) * y + accepted * v
    log_tau_y = torch.clamp(log_tau.exp() + 1e-1 * (accs - 0.65) / 50, min=1).log()
  return new_y, log_tau_y, accs


def gen_ordinal(log_tau, FLAGS, model, im_neg, num_steps, sample=False):
  y = im_neg  
  im_negs_samples = []
  for step in range(num_steps):
    s = model.forward(y, None).mean().detach()
    print(step, s.item())
    y, log_tau, accs = o_mcmc_step(y, log_tau, model)
    y = y.detach()
    print(step, accs, log_tau.item())
    if sample:
        im_negs_samples.append(y)
  if sample:
    return y, im_negs_samples, log_tau
  else:
    return y, log_tau


def get_posterior_f(z, log_tau, model, val, raw_shape):
  z.requires_grad_()
  y = torch.sum(z * val, dim=-1)
  y = y.view(raw_shape)
  logp = -torch.sum(model.forward(y, None), dim=-1, keepdims=True) * 100
  grad_z = torch.autograd.grad(logp.sum(), z)[0].detach()
  
  with torch.no_grad():
    score_change_z = torch.where(z > 0, torch.zeros_like(z), (grad_z - (grad_z * z).sum(dim=-1, keepdim=True)))
    log_weight_z = score_change_z / 2.0
    log_posterior_z = log_tau + log_weight_z
    log_posterior_z = torch.where(z > 0, torch.log1p(-torch.clamp(
        (log_posterior_z.exp() * (1 - z)).sum(-1, keepdim=True), max=1 - 1e-6)), log_posterior_z)
  return logp, log_posterior_z


def c_mcmc_step(y, log_tau, model, val):
  y_cat = (y * 256.0).to(torch.int64)
  z = F.one_hot(y_cat, num_classes=256).view(y.shape[0], -1, 256).float()
  z_rank = len(z.shape) - 1

  logp_current, log_posterior_z = get_posterior_f(z, log_tau, model, val, y_cat.shape)
  dist_current = dists.Multinomial(logits=log_posterior_z)
  w = dist_current.sample()
  logp_next, log_posterior_w = get_posterior_f(w, log_tau, model, val, y_cat.shape)
  with torch.no_grad():
    dist_next = dists.Multinomial(logits=log_posterior_w)
    log_acc = logp_next.view(-1) + dist_next.log_prob(z).sum(-1) - logp_current.view(-1) - dist_current.log_prob(w).sum(-1)
    accepted = (log_acc.exp() >= torch.rand_like(log_acc)).to(z.dtype).view(-1, *([1] * z_rank))
    new_z = w * accepted + (1.0 - accepted) * z
    accs = torch.clamp(log_acc, max=0).exp().mean().item()
    log_tau = torch.clamp(log_tau.exp() + 1e-1 * (accs - 0.65) / 50, min=1).log()
    new_z = torch.argmax(new_z, axis=-1).view(y_cat.shape).float() / 256.0    
  return new_z, log_tau, accs


def gen_categorical(log_tau, FLAGS, model, im_neg, num_steps, sample=False):
  y = im_neg  
  im_negs_samples = []
  val = torch.arange(256, dtype=torch.float32).to(y.device) / 256.0
  for step in range(num_steps):
    s = model.forward(y, None).mean().detach()
    print(step, s.item())
    y, log_tau, accs = c_mcmc_step(y, log_tau, model, val)
    y = y.detach()
    print(step, accs, log_tau.item())
    if sample:
        im_negs_samples.append(y)
  if sample:
    return y, im_negs_samples, log_tau
  else:
    return y, log_tau


if __name__ == '__main__':
  torch.manual_seed(1)
  np.random.seed(1)
  class ResNetModel(nn.Module):
    def __init__(self):
      super(ResNetModel, self).__init__()
      self.proj = nn.Linear(32, 1)

    def forward(self, x, _):
      score = self.proj(x)
      score = torch.sum(score, (1, 2))
      return score

  model = ResNetModel()
  data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (128, 32, 32, 3)))
  data_corrupt = torch.Tensor(data_corrupt.float()).permute(0, 3, 1, 2).float().contiguous()

  log_tau = torch.tensor([0], device=data_corrupt.device)
  # gen_ordinal(log_tau, None, model, data_corrupt, 10, sample=True)
  gen_categorical(log_tau, None, model, data_corrupt, 10, sample=True)
