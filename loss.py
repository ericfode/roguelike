# interestingness loss — bootstraps interesting worlds from noise
# all ops are batched tensor ops on GPU, no python loops
from tinygrad import Tensor

H, W = 24, 80

def _std(x): return ((x - x.mean())**2).mean().sqrt()

# spatial kernels
_K3 = Tensor.ones(1, 1, 3, 3) / 9.0
_SOBEL_X = Tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float().reshape(1, 1, 3, 3)
_SOBEL_Y = Tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).float().reshape(1, 1, 3, 3)

def surprise(stacked):
  """temporal change between consecutive frames. [N, H, W] -> scalar"""
  return (stacked[1:] - stacked[:-1]).square().mean().clip(0, 10)

def coherence(frame):
  """structured worlds: smooth local patches + diverse global content + clear edges"""
  x = frame.reshape(1, 1, H, W)
  avg = x.conv2d(_K3, padding=1)
  local_var = (x - avg).square().mean()
  global_var = ((frame - frame.mean()) ** 2).mean()
  gx = x.conv2d(_SOBEL_X, padding=1)
  gy = x.conv2d(_SOBEL_Y, padding=1)
  edge_mag = (gx.square() + gy.square()).sqrt().mean()
  return (global_var * 2.0 + edge_mag - local_var).clip(-1, 2)

def persistence(stacked):
  """pearson correlation between consecutive frames. [N, H, W] -> scalar"""
  a = stacked[:-1].reshape(stacked.shape[0] - 1, -1)
  b = stacked[1:].reshape(stacked.shape[0] - 1, -1)
  a_c, b_c = a - a.mean(axis=1, keepdim=True), b - b.mean(axis=1, keepdim=True)
  num = (a_c * b_c).mean(axis=1)
  den = a_c.square().mean(axis=1).sqrt() * b_c.square().mean(axis=1).sqrt() + 1e-6
  return (num / den).mean().clip(-1, 1)

def diversity(stacked):
  """tile type diversity — std across all frame values"""
  return _std(stacked.flatten()).clip(0, 1)

def temporal_complexity(stacked):
  """reward non-repeating temporal patterns — penalizes static AND periodic worlds.
  measures variance of frame-to-frame differences over time."""
  if stacked.shape[0] < 3: return Tensor(0.0)
  diffs = (stacked[1:] - stacked[:-1]).reshape(stacked.shape[0] - 1, -1)  # [N-1, H*W]
  diff_norms = diffs.square().mean(axis=1)  # [N-1] magnitude of each transition
  return _std(diff_norms).clip(0, 5)  # high = varied transitions, low = repetitive

def spatial_entropy(frame):
  """patch-level entropy — rewards worlds with distinct regions.
  divides frame into 4x4 patches, measures diversity of patch means."""
  ph, pw = 6, 20  # 24/4=6, 80/4=20 -> 4x4 grid of patches
  patches = frame.reshape(4, ph, 4, pw).permute(0, 2, 1, 3).reshape(16, ph * pw)
  patch_means = patches.mean(axis=1)  # [16] mean value per patch
  return _std(patch_means).clip(0, 1)  # high = patches are different from each other

def asymmetry(frame):
  """reward breaking symmetry — penalizes horizontally/vertically symmetric worlds"""
  h_flip = frame[:, ::-1] if hasattr(frame, '__getitem__') else frame.flip(1)
  v_flip = frame[::-1, :] if hasattr(frame, '__getitem__') else frame.flip(0)
  h_sym = 1.0 - (frame - h_flip).square().mean()
  v_sym = 1.0 - (frame - v_flip).square().mean()
  # penalize symmetry: return high when asymmetric
  return (2.0 - h_sym - v_sym).clip(0, 2)

def _entropy(stacked):
  """value distribution entropy — prevents mode collapse"""
  flat = stacked.flatten()
  p = flat.softmax()
  return -(p * (p + 1e-8).log()).sum() / stacked.shape[0]

# weights tuned for interesting emergence
W_S, W_C, W_P, W_D, W_TC, W_SE, W_A, W_E = 2.0, 2.0, 1.0, 1.5, 1.0, 1.5, 0.5, 0.02
WARMUP_COHERENCE, WARMUP_SURPRISE, WARMUP_PERSIST = 0, 50, 150

def _ramp(tick, start): return min(1.0, max(0.0, (tick - start) / 100.0)) if tick is not None else 1.0

def interestingness(frames, tick=None):
  """combined loss. MINIMIZE = MAXIMIZE interestingness.
  returns (loss, surprise, coherence, persistence)"""
  assert len(frames) >= 2
  stacked = Tensor.stack(*frames)  # [N, H, W] — single GPU op
  s = surprise(stacked)
  c = coherence(stacked[-1])
  p = persistence(stacked)
  d = diversity(stacked)
  tc = temporal_complexity(stacked)
  se = spatial_entropy(stacked[-1])
  a = asymmetry(stacked[-1])
  ws = W_S * _ramp(tick, WARMUP_SURPRISE)
  wc = W_C * _ramp(tick, WARMUP_COHERENCE)
  wp = W_P * _ramp(tick, WARMUP_PERSIST)
  loss = -(ws*s + wc*c + wp*p.relu() + W_D*d + W_TC*tc + W_SE*se + W_A*a) - W_E * _entropy(stacked)
  return loss, s, c, p
