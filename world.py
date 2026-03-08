#!/usr/bin/env python3
"""WorldNet: conv autoregressive frame predictor. the world is a forward pass."""
from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear

W, H, HIDDEN, ACTION_DIM, N_TILES = 80, 24, 32, 16, 8

class WorldNet:
  def __init__(self):
    # embedding: one-hot -> hidden
    self.embed = Tensor.kaiming_uniform(N_TILES, HIDDEN)
    # 3 conv layers, 3x3 residual
    self.w1, self.b1 = Tensor.kaiming_uniform(HIDDEN, HIDDEN, 3, 3), Tensor.zeros(HIDDEN)
    self.w2, self.b2 = Tensor.kaiming_uniform(HIDDEN, HIDDEN, 3, 3), Tensor.zeros(HIDDEN)
    self.w3, self.b3 = Tensor.kaiming_uniform(HIDDEN, HIDDEN, 3, 3), Tensor.zeros(HIDDEN)
    # FiLM conditioning: action -> scale + bias per channel (applied after conv2)
    self.film_scale = Linear(ACTION_DIM, HIDDEN)
    self.film_bias = Linear(ACTION_DIM, HIDDEN)
    # decode head: 1x1 conv -> tile logits
    self.w_dec, self.b_dec = Tensor.kaiming_uniform(N_TILES, HIDDEN, 1, 1), Tensor.zeros(N_TILES)

  def forward(self, frame, action):
    # frame: [H, W] int tile indices, action: [ACTION_DIM] float
    x = Tensor.one_hot(frame, N_TILES).float()  # [H, W, N_TILES]
    x = x @ self.embed                           # [H, W, HIDDEN]
    x = x.permute(2, 0, 1).unsqueeze(0)          # [1, HIDDEN, H, W]
    # conv stack with residuals
    r = x
    x = x.conv2d(self.w1, self.b1, padding=1).relu()
    x = x.conv2d(self.w2, self.b2, padding=1).relu()
    # FiLM: action conditions the world prediction
    gamma = self.film_scale(action.unsqueeze(0)).reshape(1, HIDDEN, 1, 1) + 1.0  # scale centered at 1
    beta = self.film_bias(action.unsqueeze(0)).reshape(1, HIDDEN, 1, 1)
    x = x * gamma + beta
    x = (x + r).conv2d(self.w3, self.b3, padding=1).relu()
    x = x + r  # second residual
    act = x    # activations before decode
    logits = x.conv2d(self.w_dec, self.b_dec)  # [1, N_TILES, H, W]
    return logits.squeeze(0).permute(1, 2, 0), act.squeeze(0)  # [H, W, N_TILES], [HIDDEN, H, W]

  def expose_activations(self, frame, action):
    _, act = self.forward(frame, action)
    return act  # [HIDDEN, H, W] — what the player net sees

  def __call__(self, frame, action): return self.forward(frame, action)

if __name__ == '__main__':
  net = WorldNet()
  frame = Tensor.randint(H, W, high=N_TILES)
  action = Tensor.randn(ACTION_DIM)
  logits, act = net(frame, action)
  assert logits.shape == (H, W, N_TILES), f"bad logits shape: {logits.shape}"
  assert act.shape == (HIDDEN, H, W), f"bad activations shape: {act.shape}"
  print(f"WorldNet ok. logits {logits.shape}, activations {act.shape}, pred tile 0,0: {logits[0,0].argmax().item()}")
