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
  return diffs.clip(0, 10)  # cap surprise to prevent gradient explosion

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
    a_c, b_c = a - a.mean(), b - b.mean()  # mean-center!
    corrs.append((a_c * b_c).mean() / (_std(a) * _std(b) + 1e-6))
  return (sum(corrs) / len(corrs)).clip(-1, 1)

def _entropy(frames):
  """entropy bonus over value distribution — prevents mode collapse"""
  stacked = frames[0].flatten()
  for f in frames[1:]: stacked = stacked.cat(f.flatten())
  # histogram-like: softmax over values to get pseudo-distribution
  p = stacked.softmax()
  return -(p * (p + 1e-8).log()).sum() / len(frames)

def interestingness(frames):
  """combined loss. MINIMIZE this = MAXIMIZE interestingness. returns (loss, s, c, p)"""
  assert len(frames) >= 2
  s, c, p = surprise(frames), coherence(frames[-1]), persistence(frames)
  # additive combination instead of multiplicative to prevent explosion
  # want: high surprise, high coherence, high persistence
  loss = -(s + c + p.relu()) - 0.01 * _entropy(frames)
  return loss, s, c, p
