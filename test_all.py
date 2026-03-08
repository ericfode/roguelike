#!/usr/bin/env python3
"""smoke tests — assert-crash, no ceremony. run: python test_all.py"""
import os, time
os.environ["NVCC_PREPEND_FLAGS"] = "-w"
from tinygrad import Tensor
import numpy as np

def test_world():
  from world import WorldNet, W, H, N_TILES, HIDDEN, ACTION_DIM
  net = WorldNet()
  frame = Tensor.randint(H, W, high=N_TILES)
  frame_oh = Tensor.one_hot(frame, N_TILES).float()
  action = Tensor.randn(ACTION_DIM)
  logits, acts = net(frame_oh, action)
  assert logits.shape == (H, W, N_TILES), f"logits {logits.shape}"
  assert acts.shape == (HIDDEN, H, W), f"acts {acts.shape}"
  max_act = 0
  for i in range(50):
    frame = logits.argmax(axis=-1).detach()
    frame_oh = Tensor.one_hot(frame, N_TILES).float()
    logits, acts = net(frame_oh, action)
    vals = acts.numpy()
    assert not np.any(np.isnan(vals)), f"NaN at tick {i}"
    max_act = max(max_act, float(np.abs(vals).max()))
  assert max_act < 100, f"activation explosion: {max_act}"
  print(f"  world: 50 ticks, max_act={max_act:.1f} ✓")

def test_player():
  from player import PlayerNet, ACTION_DIM, HIDDEN, H, W
  net = PlayerNet()
  x = Tensor.randn(1, HIDDEN, H, W)
  acts, attn = net(x)
  assert acts.shape == (ACTION_DIM,), f"acts {acts.shape}"
  assert len(attn.shape) == 2, f"attn {attn.shape}"
  vals = acts.numpy()
  assert not np.any(np.isnan(vals)), "NaN in player output"
  print(f"  player: acts={acts.shape} attn={attn.shape} ✓")

def test_loss():
  from loss import interestingness
  frames = [Tensor.rand(24, 80) for _ in range(8)]
  loss, s, c, p = interestingness(frames)
  assert loss.shape == (), f"loss shape {loss.shape}"
  lv = float(loss.numpy())
  assert np.isfinite(lv), f"loss not finite: {lv}"
  print(f"  loss: val={lv:.4f} s={float(s.numpy()):.3f} c={float(c.numpy()):.3f} p={float(p.numpy()):.3f} ✓")

def test_tui():
  from tui import downsample, bar, heatmap_char
  arr = np.random.rand(48, 160)
  ds = downsample(arr, 12, 32)
  assert ds.shape == (12, 32), f"downsample {ds.shape}"
  assert heatmap_char(0.0) == ' '
  assert heatmap_char(1.0) == '█'
  b = bar(np.array([0.0, 0.5, 1.0]), 3)
  assert len(b) == 3
  print(f"  tui: downsample={ds.shape} bar='{b}' ✓")

def test_integration():
  """full loop: 20 ticks with JIT, detach, grad clip — mirrors main.py"""
  from world import WorldNet, W, H, N_TILES, HIDDEN, ACTION_DIM
  from player import PlayerNet
  from loss import interestingness
  from tinygrad.nn.optim import Adam
  from tinygrad.nn.state import get_parameters

  Tensor.training = True
  world, player = WorldNet(), PlayerNet()
  all_params = get_parameters(world) + get_parameters(player)
  opt = Adam(all_params, lr=3e-4)
  frame = Tensor.randint(H, W, high=N_TILES)
  activations = Tensor.zeros(1, HIDDEN, H, W)
  frame_buf = []
  lv = 0.0
  times = []

  for tick in range(20):
    t0 = time.monotonic()
    activations = activations.detach()
    frame_oh = Tensor.one_hot(frame, N_TILES).float()
    action, _ = player(activations)
    logits, acts_raw = world(frame_oh, action)
    activations = acts_raw.unsqueeze(0)
    tau = max(0.5, 2.0 - tick * 0.001)
    gumbel = -(-Tensor.rand(*logits.shape).clip(1e-8, 1).log()).log()
    soft = ((logits + gumbel) / tau).softmax(axis=-1)
    tile_vals = Tensor.arange(N_TILES).float() / N_TILES
    next_soft = (soft * tile_vals.reshape(1, 1, N_TILES)).sum(axis=-1)
    frame_buf.append(next_soft)
    if len(frame_buf) > 8: frame_buf = frame_buf[-8:]
    frame = logits.argmax(axis=-1).detach()

    if len(frame_buf) >= 4:
      loss, s, c, p = interestingness(frame_buf, tick=tick)
      opt.zero_grad()
      loss.backward()
      for pr in all_params:
        if pr.grad is not None: pr.grad = pr.grad.clip(-1.0, 1.0)
      opt.step()
      for pr in all_params: pr.realize()
      lv = float(loss.numpy())
      assert np.isfinite(lv), f"loss not finite at tick {tick}: {lv}"
      frame_buf = [f.detach() for f in frame_buf[:-2]] + frame_buf[-2:]

    assert not np.any(np.isnan(frame.numpy())), f"NaN frame at tick {tick}"
    times.append(time.monotonic() - t0)

  # verify no slowdown: last 5 ticks should not be >3x slower than first 5
  early, late = np.mean(times[5:10]), np.mean(times[15:20])
  ratio = late / (early + 1e-6)
  print(f"  integration: 20 ticks, loss={lv:.4f}, early={early:.1f}s late={late:.1f}s ratio={ratio:.2f}x ✓")
  assert ratio < 3.0, f"slowdown detected: {ratio:.1f}x"

if __name__ == '__main__':
  tests = [('world', test_world), ('player', test_player), ('loss', test_loss), ('tui', test_tui), ('integration', test_integration)]
  t0 = time.monotonic()
  passed, failed = 0, 0
  for name, fn in tests:
    try: fn(); passed += 1
    except Exception as e: print(f"  {name}: FAIL — {e}"); failed += 1
  dt = time.monotonic() - t0
  print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {passed}/{len(tests)} in {dt:.1f}s")
  exit(1 if failed else 0)
