#!/usr/bin/env python3
"""the game trains itself while you watch. every tick is inference AND a training step."""
import sys, os, time, signal
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from world import WorldNet, W, H, N_TILES, HIDDEN, ACTION_DIM
from player import PlayerNet
from loss import interestingness
from tui import render, init_screen, cleanup_screen

FRAME_BUF = 16
LR = 3e-4
FPS_TARGET = 10

def count_params(model): return sum(p.numel() for p in get_parameters(model))

def main():
  Tensor.training = True
  world, player = WorldNet(), PlayerNet()
  all_params = get_parameters(world) + get_parameters(player)
  opt = Adam(all_params, lr=LR)
  n_params = count_params(world) + count_params(player)

  # bootstrap: random initial frame
  frame = Tensor.randint(H, W, high=N_TILES)
  activations = Tensor.zeros(1, HIDDEN, H, W)
  last_action = Tensor.zeros(ACTION_DIM)
  frame_buf = []
  tick = 0
  fps_t = time.monotonic()

  init_screen()
  signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

  try:
    while True:
      loop_start = time.monotonic()

      # player observes world's latent state, produces action
      action, attn = player(activations)
      last_action = action

      # world takes current frame + action, predicts next frame
      logits, activations_raw = world(frame, action)
      activations = activations_raw.unsqueeze(0)  # [1, HIDDEN, H, W] for player

      # gumbel softmax for differentiable sampling
      tau = max(0.5, 2.0 - tick * 0.001)
      gumbel = -(-Tensor.rand(*logits.shape).clip(1e-8, 1).log()).log()
      soft = ((logits + gumbel) / tau).softmax(axis=-1)  # [H, W, N_TILES]
      # weighted sum over tile types -> continuous frame value
      tile_vals = Tensor.arange(N_TILES).float() / N_TILES
      next_frame_soft = (soft * tile_vals.reshape(1, 1, N_TILES)).sum(axis=-1)  # [H, W]

      frame_buf.append(next_frame_soft)
      if len(frame_buf) > FRAME_BUF: frame_buf = frame_buf[-FRAME_BUF:]

      # hard frame for next tick input (argmax, no grad)
      frame = logits.argmax(axis=-1).detach()

      # compute loss and train
      metrics = {'tick': tick, 'surprise': 0.0, 'coherence': 0.0, 'persistence': 0.0, 'loss': 0.0, 'lr': LR, 'params': n_params / 1000, 'fps': 0.0, 'tau': tau}
      if len(frame_buf) >= 4:
        buf = frame_buf[-8:] if len(frame_buf) >= 8 else frame_buf
        loss_val, s, c, p = interestingness(buf)
        opt.zero_grad()
        loss_val.backward()
        opt.step()
        metrics.update({'surprise': float(s.numpy()), 'coherence': float(c.numpy()), 'persistence': float(p.numpy()), 'loss': float(loss_val.numpy())})

      # render
      now = time.monotonic()
      metrics['fps'] = 1.0 / max(0.001, now - fps_t)
      fps_t = now
      frame_np = frame.numpy().astype(int)
      acts_raw = activations_raw.mean(axis=0).numpy()  # [H, W]
      acts_np = (acts_raw - acts_raw.min()) / (acts_raw.max() - acts_raw.min() + 1e-8)  # normalize to [0,1]
      action_np = last_action.detach().numpy().flatten()
      render(frame_np, acts_np, action_np, metrics)
      tick += 1

      # fps limiter
      elapsed = time.monotonic() - loop_start
      if elapsed < 1.0 / FPS_TARGET: time.sleep(1.0 / FPS_TARGET - elapsed)

  except (KeyboardInterrupt, SystemExit): pass
  finally:
    cleanup_screen()
    print(f"dissolved after {tick} ticks. the tensors rest.")

if __name__ == '__main__': main()
