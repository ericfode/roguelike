#!/usr/bin/env python3
"""WorldNet: conv autoregressive frame predictor. the world is a forward pass."""
from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear

W, H, HIDDEN, ACTION_DIM, N_TILES = 80, 24, 32, 16, 8

def layernorm(x, eps=1e-5):
  """channel-wise layer norm for [1, C, H, W] tensors"""
  mean = x.mean(axis=1, keepdim=True)
  var = ((x - mean) ** 2).mean(axis=1, keepdim=True)
  return (x - mean) / (var + eps).sqrt()

class WorldNet:
  def __init__(self):
    self.embed = Tensor.kaiming_uniform(N_TILES, HIDDEN)
    self.w1, self.b1 = Tensor.kaiming_uniform(HIDDEN, HIDDEN, 3, 3), Tensor.zeros(HIDDEN)
    self.w2, self.b2 = Tensor.kaiming_uniform(HIDDEN, HIDDEN, 3, 3), Tensor.zeros(HIDDEN)
    self.w3, self.b3 = Tensor.kaiming_uniform(HIDDEN, HIDDEN, 3, 3), Tensor.zeros(HIDDEN)
    # learnable layer norm scales
    self.ln1_g, self.ln1_b = Tensor.ones(1, HIDDEN, 1, 1), Tensor.zeros(1, HIDDEN, 1, 1)
    self.ln2_g, self.ln2_b = Tensor.ones(1, HIDDEN, 1, 1), Tensor.zeros(1, HIDDEN, 1, 1)
    self.film_scale, self.film_bias = Linear(ACTION_DIM, HIDDEN), Linear(ACTION_DIM, HIDDEN)
    self.w_dec, self.b_dec = Tensor.kaiming_uniform(N_TILES, HIDDEN, 1, 1), Tensor.zeros(N_TILES)
    self.last_activations = None

  def forward(self, frame, action):
    x = Tensor.one_hot(frame, N_TILES).float() @ self.embed  # [H, W, HIDDEN]
    x = x.permute(2, 0, 1).unsqueeze(0)                      # [1, HIDDEN, H, W]
    r = x
    x = x.conv2d(self.w1, self.b1, padding=1).relu()
    x = x.conv2d(self.w2, self.b2, padding=1).relu()
    # FiLM conditioning
    gamma = self.film_scale(action.unsqueeze(0)).reshape(1, HIDDEN, 1, 1) + 1.0
    beta = self.film_bias(action.unsqueeze(0)).reshape(1, HIDDEN, 1, 1)
    x = x * gamma + beta
    # residual with layer norm — prevents activation explosion
    x = layernorm(x + r) * self.ln1_g + self.ln1_b
    x = x.conv2d(self.w3, self.b3, padding=1).relu()
    x = layernorm(x + r) * self.ln2_g + self.ln2_b
    self.last_activations = x
    logits = x.conv2d(self.w_dec, self.b_dec)
    return logits.squeeze(0).permute(1, 2, 0), x.squeeze(0)  # [H, W, N_TILES], [HIDDEN, H, W]

  def __call__(self, frame, action): return self.forward(frame, action)
