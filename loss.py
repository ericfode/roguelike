# interestingness loss for tensor roguelike — bootstraps interesting worlds from noise
from tinygrad import Tensor

H, W = 24, 80

def _std(x): return ((x - x.mean())**2).mean().sqrt()

# 3x3 averaging kernel for spatial coherence, [1,1,3,3] for conv2d
_K = Tensor.ones(1, 1, 3, 3) / 9.0

def surprise(frames):
  """higher = more change between frames = good. clamped to prevent explosion"""
  assert len(frames) >= 2
  diffs = sum((frames[i+1] - frames[i]).square().mean() for i in range(len(frames)-1)) / (len(frames)-1)
  return diffs.clip(0, 10)

def coherence(frame):
  """higher = more spatial structure = good. frame is [H,W] float"""
  x = frame.reshape(1, 1, H, W)
  avg = x.conv2d(_K, padding=1)
  return (1.0 - (x - avg).square().mean()).clip(0, 1)

def persistence(frames):
  """pearson correlation between consecutive frames. properly mean-centered."""
  assert len(frames) >= 2
  corrs = []
  for i in range(len(frames)-1):
    a, b = frames[i].flatten(), frames[i+1].flatten()
    a_c, b_c = a - a.mean(), b - b.mean()
    corrs.append((a_c * b_c).mean() / (_std(a) * _std(b) + 1e-6))
  return (sum(corrs) / len(corrs)).clip(-1, 1)

def diversity(frames):
  """encourages using full value range — penalizes mode collapse to single tile type"""
  stacked = frames[0].flatten()
  for f in frames[1:]: stacked = stacked.cat(f.flatten())
  return _std(stacked).clip(0, 1)

def _entropy(frames):
  """entropy bonus over value distribution — prevents mode collapse"""
  stacked = frames[0].flatten()
  for f in frames[1:]: stacked = stacked.cat(f.flatten())
  p = stacked.softmax()
  return -(p * (p + 1e-8).log()).sum() / len(frames)

# base weights
W_S, W_C, W_P, W_D, W_E = 2.0, 1.0, 1.0, 0.5, 0.01
# curriculum: ticks at which each objective reaches full weight
WARMUP_COHERENCE, WARMUP_SURPRISE, WARMUP_PERSIST = 0, 100, 200

def _ramp(tick, start): return min(1.0, max(0.0, (tick - start) / 100.0)) if tick is not None else 1.0

def interestingness(frames, tick=None):
  """combined loss. MINIMIZE this = MAXIMIZE interestingness. returns (loss, s, c, p)
  tick: optional, enables curriculum warmup — coherence first, then surprise, then persistence"""
  assert len(frames) >= 2
  s, c, p = surprise(frames), coherence(frames[-1]), persistence(frames)
  d = diversity(frames)
  ws = W_S * _ramp(tick, WARMUP_SURPRISE)
  wc = W_C * _ramp(tick, WARMUP_COHERENCE)
  wp = W_P * _ramp(tick, WARMUP_PERSIST)
  loss = -(ws * s + wc * c + wp * p.relu() + W_D * d) - W_E * _entropy(frames)
  return loss, s, c, p
