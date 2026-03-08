import sys, numpy as np

W, H, PANEL_W, N_TILES = 80, 24, 36, 8
GLYPHS = {0: '.', 1: '#', 2: '>', 3: '@', 4: 'g', 5: 'o', 6: 'D', 7: 'T'}
COLORS = {0: '90', 1: '37', 2: '33', 3: '32;1', 4: '31', 5: '31;1', 6: '35;1', 7: '36'}
HEAT_CHARS = ' ░▒▓█'
SPARK = '▁▂▃▄▅▆▇█'
HEAT_COLORS = ['34', '36', '32', '33', '31']  # blue cyan green yellow red

def heatmap_char(v): return HEAT_CHARS[min(int(np.clip(v, 0, 1) * len(HEAT_CHARS)), len(HEAT_CHARS) - 1)]
def heatmap_color(v): return HEAT_COLORS[min(int(np.clip(v, 0, 1) * len(HEAT_COLORS)), len(HEAT_COLORS) - 1)]
def bar(values, width): v = np.clip(np.asarray(values), 0, 1); idx = np.minimum((v * len(SPARK)).astype(int), len(SPARK) - 1); return ''.join(SPARK[i] for i in idx[:width])

def downsample(arr, th, tw):
  h, w = arr.shape
  bh, bw = max(1, h // th), max(1, w // tw)
  th2, tw2 = min(th, h), min(tw, w)
  return np.array([[arr[r*bh:min((r+1)*bh, h), c*bw:min((c+1)*bw, w)].mean() for c in range(tw2)] for r in range(th2)])

def _tile_line(row):
  return ''.join(f'\033[{COLORS.get(int(v), "37")}m{GLYPHS.get(int(v), "?")}' for v in row) + '\033[0m'

def _heat_line(row):
  return ''.join(f'\033[{heatmap_color(v)}m{heatmap_char(v)}' for v in row) + '\033[0m'

def _pad(s, w):
  # strip ANSI to count visible length, then pad
  import re
  vis = len(re.sub(r'\033\[[^m]*m', '', s))
  return s + ' ' * max(0, w - vis)

def _box_top(label, w): t = f'─ {label} '; return '┌' + t + '─' * max(0, w - 2 - len(t)) + '┐'
def _box_bot(w): return '└' + '─' * (w - 2) + '┘'
def _box_mid(label, w): t = f'─ {label} '; return '├' + t + '─' * max(0, w - 2 - len(t)) + '┤'
def _box_row(content, w): return '│' + _pad(content, w - 2) + '│'

def render(frame, activations, action, metrics):
  assert frame.shape == (H, W), f"frame shape {frame.shape} != ({H},{W})"
  buf = []
  out = lambda s: buf.append(s)
  heat = downsample(np.clip(activations, 0, 1), 12, PANEL_W - 4) if activations is not None else np.zeros((12, PANEL_W - 4))
  act = np.asarray(action) if action is not None else np.zeros(8)
  act_norm = np.clip((act - act.min()) / (act.max() - act.min() + 1e-8), 0, 1)
  conf = float(np.max(act_norm))
  entropy = float(-np.sum(np.clip(act_norm / (act_norm.sum() + 1e-8), 1e-8, 1) * np.log(np.clip(act_norm / (act_norm.sum() + 1e-8), 1e-8, 1))))
  m = metrics or {}

  out('\033[H')  # home cursor, no clear
  # top borders
  out(_box_top('WORLD', W + 2) + _box_top('ACTIVATIONS', PANEL_W + 2) + '\n')
  # world rows + right panel
  heat_h = heat.shape[0]
  player_start = max(12, heat_h) + 1  # where player net panel starts
  for r in range(H):
    left = _box_row(_tile_line(frame[r]), W + 2)
    if r < heat_h:
      right = _box_row(_heat_line(heat[r]), PANEL_W + 2)
    elif r == heat_h:
      right = _box_mid('PLAYER NET', PANEL_W + 2)
    elif r == heat_h + 1:
      right = _box_row(f'action: [{",".join(f"{v:.1f}" for v in act[:8])}]', PANEL_W + 2)
    elif r == heat_h + 2:
      right = _box_row(f'confidence: {conf:.2f}', PANEL_W + 2)
    elif r == heat_h + 3:
      right = _box_row(bar(act_norm, min(len(act_norm), PANEL_W - 4)), PANEL_W + 2)
    elif r == heat_h + 4:
      right = _box_row(f'entropy: {entropy:.2f}', PANEL_W + 2)
    elif r == H - 1:
      right = _box_bot(PANEL_W + 2)
    else:
      right = _box_row('', PANEL_W + 2)
    out(left + right + '\n')
  # metrics bar
  ml = f"tick:{m.get('tick',0)} surprise:{m.get('surprise',0):.2f} coher:{m.get('coherence',0):.2f} pers:{m.get('persistence',0):.2f}"
  mr = f"loss:{m.get('loss',0):.4f} lr:{m.get('lr',0):.0e} params:{m.get('params',0):.1f}K fps:{m.get('fps',0):.1f}"
  metrics_line = f'{ml}  {mr}'
  out(_box_mid('METRICS', W + 2) + '\n')
  out(_box_row(metrics_line, W + 2) + '\n')
  out(_box_bot(W + 2) + '\n')
  sys.stdout.write(''.join(buf))
  sys.stdout.flush()

def init_screen(): sys.stdout.write('\033[2J\033[H\033[?25l'); sys.stdout.flush()
def cleanup_screen(): sys.stdout.write('\033[?25h\033[0m\n'); sys.stdout.flush()

if __name__ == '__main__':
  init_screen()
  try:
    frame = np.random.randint(0, N_TILES, (H, W))
    frame[frame > 3] = np.random.choice([0, 1], size=(frame > 3).sum(), p=[0.7, 0.3])  # mostly floor/wall
    frame[12, 40] = 3  # player
    for _ in range(8): frame[np.random.randint(H), np.random.randint(W)] = np.random.randint(4, 8)
    acts = np.random.rand(H * 2, W * 2) * np.random.rand(H * 2, W * 2)  # sparse activations
    action = np.random.randn(8)
    metrics = dict(tick=1847, surprise=0.34, coherence=0.91, persistence=0.88, loss=0.027, lr=3e-4, params=48.2, fps=12.3)
    render(frame, acts, action, metrics)
    import time; time.sleep(3)
  finally:
    cleanup_screen()
