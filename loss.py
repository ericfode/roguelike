# interestingness loss — bootstraps interesting worlds from noise
# all ops are batched tensor ops, no python loops in the hot path
from tinygrad import Tensor

H, W = 24, 80

def _std(x): return ((x - x.mean())**2).mean().sqrt()

# kernels for spatial ops
_K3 = Tensor.ones(1, 1, 3, 3) / 9.0   # 3x3 avg
_SOBEL_X = Tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float().reshape(1, 1, 3, 3)
_SOBEL_Y = Tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).float().reshape(1, 1, 3, 3)

def surprise(stacked):
  """batch surprise: mean squared diff between consecutive frames. stacked: [N, H, W]"""
  diffs = (stacked[1:] - stacked[:-1]).square().mean(axis=(1, 2))  # [N-1]
  return diffs.mean().clip(0, 10)

def coherence(frame):
  """rewards structured worlds — penalizes both noise AND uniformity.
  high score = smooth local patches + diverse global content (like a real map)"""
  x = frame.reshape(1, 1, H, W)
  # local smoothness: low = good (neighbors are similar)
  avg = x.conv2d(_K3, padding=1)
  local_var = (x - avg).square().mean()
  # global diversity: high = good (not all the same tile)
  global_var = ((frame - frame.mean()) ** 2).mean()
  # edge structure: reward having clear boundaries (sobel magnitude in sweet spot)
  gx = x.conv2d(_SOBEL_X, padding=1)
  gy = x.conv2d(_SOBEL_Y, padding=1)
  edge_mag = (gx.square() + gy.square()).sqrt().mean()
  # want: high global diversity + moderate edges - low local noise
  return (global_var * 2.0 + edge_mag - local_var).clip(-1, 2)

def persistence(stacked):
  """batch pearson correlation between consecutive frames. stacked: [N, H, W]"""
  a = stacked[:-1].reshape(stacked.shape[0] - 1, -1)  # [N-1, H*W]
  b = stacked[1:].reshape(stacked.shape[0] - 1, -1)    # [N-1, H*W]
  a_c, b_c = a - a.mean(axis=1, keepdim=True), b - b.mean(axis=1, keepdim=True)
  num = (a_c * b_c).mean(axis=1)  # [N-1]
  den = a_c.square().mean(axis=1).sqrt() * b_c.square().mean(axis=1).sqrt() + 1e-6
  return (num / den).mean().clip(-1, 1)

def diversity(stacked):
  """tile type diversity across all frames"""
  return _std(stacked.flatten()).clip(0, 1)

def _entropy(stacked):
  """entropy bonus — prevents mode collapse"""
  flat = stacked.flatten()
  p = flat.softmax()
  return -(p * (p + 1e-8).log()).sum() / stacked.shape[0]

# weights
W_S, W_C, W_P, W_D, W_E = 2.0, 2.0, 1.0, 1.0, 0.02
WARMUP_COHERENCE, WARMUP_SURPRISE, WARMUP_PERSIST = 0, 50, 150

def _ramp(tick, start): return min(1.0, max(0.0, (tick - start) / 100.0)) if tick is not None else 1.0

def interestingness(frames, tick=None):
  """combined loss. all frames stacked into single tensor for GPU batching.
  returns (loss, surprise, coherence, persistence)"""
  assert len(frames) >= 2
  stacked = frames[0].unsqueeze(0)
  for f in frames[1:]: stacked = stacked.cat(f.unsqueeze(0), dim=0)  # [N, H, W]
  s = surprise(stacked)
  c = coherence(stacked[-1])
  p = persistence(stacked)
  d = diversity(stacked)
  ws = W_S * _ramp(tick, WARMUP_SURPRISE)
  wc = W_C * _ramp(tick, WARMUP_COHERENCE)
  wp = W_P * _ramp(tick, WARMUP_PERSIST)
  loss = -(ws * s + wc * c + wp * p.relu() + W_D * d) - W_E * _entropy(stacked)
  return loss, s, c, p
