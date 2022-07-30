import torch
import torch.distributions as dists
import torch.nn as nn
import numpy as np


def get_posterior(y, log_tau, model):  
  y.requires_grad_()
  logp = -torch.sum(model.forward(y, None), dim=-1, keepdims=True)
  grad_y = torch.autograd.grad(logp.sum(), y)[0].detach()
  delta = torch.arange(256, dtype=torch.float32).to(y.device) / 256.0 - torch.unsqueeze(y, -1)
  grad_y = torch.unsqueeze(grad_y, -1)
  rate_y = delta * grad_y - delta ** 2 / torch.exp(log_tau) / 2
  log_posterior_y = rate_y / 2.0
  return logp, log_posterior_y


def mcmc_step(y, log_tau, model):
  logp_current, log_posterior_y = get_posterior(y, log_tau, model)
  dist_y = dists.Categorical(logits=log_posterior_y)
  v_cat = dist_y.sample()
  v = v_cat.float() / 256.0

  logp_next, log_posterior_v = get_posterior(v, log_tau, model)
  dist_v = dists.Categorical(logits=log_posterior_v)
  y_cat = (y * 256.0).to(torch.int64)
  log_forward = torch.sum(dist_y.log_prob(v_cat), (1, 2, 3)) + logp_current.view(-1)
  log_backward = torch.sum(dist_v.log_prob(y_cat), (1, 2, 3)) + logp_next.view(-1)
  log_acc = log_backward - log_forward
  accepted = (torch.rand_like(log_acc) < log_acc.exp()).float().view(-1, 1, 1, 1)
  accs = torch.clamp(log_acc, max=0).exp().mean().item()
  new_y = (1.0 - accepted) * y + accepted * v
  log_tau_y = torch.clamp(log_tau.exp() + 1e-1 * (accs - 0.65) / 50, min=1e-10).log()
  return new_y, log_tau_y


def gen_ordinal(log_tau, FLAGS, model, im_neg, num_steps, sample=False):
  y = im_neg  
  im_negs_samples = []
  for step in range(num_steps):
    y, log_tau = mcmc_step(y, log_tau, model)
    s = model.forward(y, None).mean()
    print(step, s.item())
    if sample:
        im_negs_samples.append(y)
  if sample:
    return y, im_negs_samples, log_tau
  else:
    return y, log_tau


if __name__ == '__main__':
  class ResNetModel(nn.Module):
    def __init__(self):
      super(ResNetModel, self).__init__()
      self.proj = nn.Linear(32, 1)

    def forward(self, x):
      score = self.proj(x)
      score = torch.sum(score, (1, 2))
      return score

  model = ResNetModel()
  data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (128, 32, 32, 3)))
  data_corrupt = torch.Tensor(data_corrupt.float()).permute(0, 3, 1, 2).float().contiguous()

  log_tau = torch.tensor([0], device=data_corrupt.device)
  gen_ordinal(log_tau, None, model, data_corrupt, 10, sample=True)
