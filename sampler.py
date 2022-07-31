import torch
import torch.distributions as dists
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_posterior_o(y, log_tau, model, alpha):  
  y.requires_grad_()
  logp = -torch.sum(model.forward(y, None), dim=-1, keepdims=True) * 100000
  grad_y = torch.autograd.grad(logp.sum(), y)[0].detach()
  delta = torch.arange(256, dtype=torch.float32).to(y.device) / 256.0 - torch.unsqueeze(y, -1)
  grad_y = torch.unsqueeze(grad_y, -1)
  alpha = alpha.unsqueeze(-1).unsqueeze(0) * 0 + 1.0
  rate_y = delta * grad_y - alpha * delta ** 2 / torch.exp(log_tau) / 2
  log_posterior_y = rate_y / 2.0
  return logp, log_posterior_y, grad_y.squeeze(-1)


def o_mcmc_step(y, log_tau, model):
  log_tau, grad2, grad = log_tau
  logp_current, log_posterior_y, grad_y = get_posterior_o(y, log_tau, model, grad2 / grad)
  dist_y = dists.Categorical(logits=log_posterior_y)
  v_cat = dist_y.sample()
  v = v_cat.float() / 256.0
  logp_next, log_posterior_v, grad_v = get_posterior_o(v, log_tau, model, grad2 / grad)
  with torch.no_grad():
    dist_v = dists.Categorical(logits=log_posterior_v)
    y_cat = (y * 256.0).to(torch.int64)
    log_forward = torch.sum(dist_y.log_prob(v_cat), (1, 2, 3)) + logp_current.view(-1)
    log_backward = torch.sum(dist_v.log_prob(y_cat), (1, 2, 3)) + logp_next.view(-1)
    #log_acc = log_backward - log_forward
    log_acc = logp_next.view(-1) - logp_current.view(-1)
    accepted = (torch.rand_like(log_acc) < log_acc.exp()).float().view(-1, 1, 1, 1)
    accs = torch.clamp(log_acc, max=0).exp().mean().item()
    new_y = (1.0 - accepted) * y + accepted * v
    log_tau_y = torch.clamp(log_tau.exp() + 1e-1 * (accs - 0.65) / 50, min=1e-10).log()
    grad2 = 0.9 * grad2 + 0.1 * (grad_v - grad_y).abs().mean(0)
    grad = 0.9 * grad + 0.1 * torch.clamp((v - y).abs().mean(), min=1.0 / 256.0)
  return new_y, (log_tau_y, grad2, grad), accs


def gen_ordinal(log_tau, FLAGS, model, im_neg, num_steps, sample=False):
  y = im_neg  
  im_negs_samples = []
  for step in range(num_steps):
    y = torch.clamp(y, max=0.999)
    y, log_tau, accs = o_mcmc_step(y, log_tau, model)
    y = y.detach()
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
  logp = -torch.sum(model.forward(y, None), dim=-1, keepdims=True) * 500000
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
    #log_acc = logp_next.view(-1) + dist_next.log_prob(z).sum(-1) - logp_current.view(-1) - dist_current.log_prob(w).sum(-1)
    log_acc = logp_next.view(-1) - logp_current.view(-1) 
    accepted = (log_acc.exp() >= torch.rand_like(log_acc)).to(z.dtype).view(-1, *([1] * z_rank))
    new_z = w * accepted + (1.0 - accepted) * z
    accs = torch.clamp(log_acc, max=0).exp().mean().item()
    log_tau = torch.clamp(log_tau.exp() + 1e-1 * (accs - 0.65) / 50, min=1).log()
    new_z = torch.argmax(new_z, axis=-1).view(y_cat.shape).float() / 256.0
  return new_z, log_tau, accs


def gen_categorical(log_tau, FLAGS, model, im_neg, num_steps, sample=False):
  y = torch.clamp(im_neg, max=0.999)
  im_negs_samples = []
  val = torch.arange(256, dtype=torch.float32).to(y.device) / 256.0
  s = model.forward(y, None).mean().detach()
  print(s.item())
  for step in range(num_steps):
    y, log_tau, accs = c_mcmc_step(y, log_tau, model, val)
    y = y.detach()
    if sample:
        im_negs_samples.append(y)
  s = model.forward(y, None).mean().detach()
  print(s.item())
  if sample:
    return y, im_negs_samples, log_tau
  else:
    return y, log_tau


def get_score_grad(z, model, val, raw_shape):
  z.requires_grad_()
  y = torch.sum(z * val, dim=-1)
  y = y.view(raw_shape)
  logp = -torch.sum(model.forward(y, None), dim=-1, keepdims=True) * 10000
  grad_z = torch.autograd.grad(logp.sum(), z)[0].detach()
  with torch.no_grad():
    score_change_z = 0.5 * (grad_z - (grad_z * z).sum(dim=-1, keepdim=True))
    score_change_z = score_change_z - 1e9 * z
    dist = dists.Categorical(logits=score_change_z)
  return logp, grad_z, dist


def gwg_mcmc_step(y_init, model, val):
  y_cat = (y_init * 256.0).to(torch.int64)
  x = F.one_hot(y_cat, num_classes=256).view(y_init.shape[0], -1, 256).float()
  score_x, _, dist_x = get_score_grad(x, model, val, y_cat.shape)

  index_x = dist_x.sample()
  log_x2y = torch.sum(dist_x.log_prob(index_x), dim=1, keepdims=True)

  y = F.one_hot(index_x, num_classes=256).view(y_cat.shape[0], -1, 256).float()
  score_y, _, dist_y = get_score_grad(y, model, val, y_cat.shape)

  with torch.no_grad():
    log_y2x = torch.sum(dist_y.log_prob(y_cat.view(y.shape[0], -1)), dim=1, keepdims=True)  
    log_acc = score_y + log_y2x - score_x - log_x2y
    accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, 1, 1)
    new_x = y * accepted + (1.0 - accepted) * x
    accs = torch.clamp(log_acc.exp(), max=1).mean().item()
    new_x = torch.argmax(new_x, axis=-1).view(y_init.shape).float() / 256.0
  return new_x, accs


def gen_gwg(log_tau, FLAGS, model, im_neg, num_steps, sample=False):
  y = torch.clamp(im_neg, max=0.999)
  im_negs_samples = []
  val = torch.arange(256, dtype=torch.float32).to(y.device) / 256.0
  for step in range(num_steps):
    y, accs = gwg_mcmc_step(y, model, val)
    y = y.detach()
    if sample:
        im_negs_samples.append(y)    
  if sample:
    return y, im_negs_samples, log_tau
  else:
    return y, log_tau

def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def pafs_step(x, model):
    bsize = x.shape[0]
    b_idx = torch.arange(bsize).unsqueeze(-1)
    x_rank = len(x.shape) - 1
    R = 1000

    x.requires_grad_()
    score_x = model(x)
    grad_x = torch.autograd.grad(score_x.sum(), x)[0].detach()
    with torch.no_grad():
        score_change_x = (grad_x - (grad_x * x).sum(dim=-1, keepdim=True)) / 2.0
        score_change_x[x>0] = -float("inf")
        log_prob_x_site = torch.logsumexp(score_change_x, dim=2) - \
                      torch.logsumexp(score_change_x, dim=[1, 2]).unsqueeze(-1)
        index = torch.multinomial(log_prob_x_site.exp(), R)
        log_prob_x_value = score_change_x[b_idx, index]
        dist_x = dists.Multinomial(logits=log_prob_x_value)
        y_value = dist_x.sample()
        y = x.clone()
        y[b_idx, index] = y_value

    y.requires_grad_()
    score_y = model(y)
    grad_y = torch.autograd.grad(score_y.sum(), y)[0].detach()
    with torch.no_grad():
        score_change_y = (grad_y - (grad_y * y).sum(dim=-1, keepdim=True)) / 2.0
        score_change_y[y > 0] = -float("inf")
        log_prob_y_site = torch.logsumexp(score_change_y, dim=2) - \
                          torch.logsumexp(score_change_y, dim=[1, 2]).unsqueeze(-1)
        log_prob_y_value = score_change_y[b_idx, index]
        dist_y = dists.Multinomial(logits=log_prob_y_value)
        x_value = x[b_idx, index]
        tri_u = torch.triu(torch.ones(R, R, device=x.device), 1)
        log_x_selected = log_prob_x_site[b_idx, index]
        log_x_max = torch.max(log_x_selected, dim=-1, keepdim=True).values
        log_x_u = log_x_max + torch.log(torch.exp(log_x_selected - log_x_max) @ tri_u)
        log_x = (log_x_selected - log1mexp(log_x_u)).sum(dim=-1)
        # log_x = (log_x_selected - torch.log1p(-torch.exp(log_x_u))).sum(dim=-1)

        tri_l = torch.tril(torch.ones(R, R, device=x.device), -1)
        log_y_selected = log_prob_y_site[b_idx, index]
        log_y_max = torch.max(log_y_selected, dim=-1, keepdim=True).values
        log_y_l = log_y_max + torch.log(torch.exp(log_y_selected - log_y_max) @ tri_l)
        log_y = (log_y_selected - log1mexp(log_y_l)).sum(dim=-1)
        # log_y = (log_y_selected - torch.log1p(-torch.exp(log_y_l))).sum(dim=-1)

        log_acc = score_y + log_y + dist_y.log_prob(x_value).sum(-1) - \
                  score_x - log_x - dist_x.log_prob(y_value).sum(-1)
        accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
        new_x = y * accepted + (1.0 - accepted) * x

    accs = torch.clamp(log_acc.exp(), max=1).mean().item()
    return new_x, accs


def gen_pafs(log_tau, FLAGS, model, im_neg, num_steps, sample=False):
  val = torch.arange(256, dtype=torch.float32).to(im_neg.device) / 256.0
  model_fn = lambda x: -torch.sum(model.forward(torch.sum(x * val, dim=-1).view(im_neg.shape), None), dim=-1) * 10000
  y_cat = (torch.clamp(im_neg, max=0.999) * 256.0).to(torch.int64)
  y = F.one_hot(y_cat, num_classes=256).view(im_neg.shape[0], -1, 256).float()

  im_negs_samples = []
  val = torch.arange(256, dtype=torch.float32).to(y.device) / 256.0
  for step in range(num_steps):
    s = model.forward(torch.sum(y * val, dim=-1).view(im_neg.shape), None).mean().detach()
    print(s.item())
    y, accs = pafs_step(y, model_fn)
    y = y.detach()
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
  gen_pafs(log_tau, None, model, data_corrupt, 10, sample=True)
  # gen_categorical(log_tau, None, model, data_corrupt, 10, sample=True)
