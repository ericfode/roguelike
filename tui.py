import sys, curses, numpy as np

W, H, PANEL_W, N_TILES = 80, 24, 36, 8
GLYPHS = {0: '.', 1: '#', 2: '>', 3: '@', 4: 'g', 5: 'o', 6: 'D', 7: 'T'}
HEAT_CHARS = ' ░▒▓█'
SPARK = '▁▂▃▄▅▆▇█'

# curses color pairs: 1-8 tile colors, 10-14 heat colors
_scr = None

def _init_colors():
  curses.start_color()
  curses.use_default_colors()
  # tile colors: 0=dark gray, 1=white, 2=yellow, 3=green, 4=red, 5=red, 6=magenta, 7=cyan
  tile_colors = [8, 7, 3, 2, 1, 1, 5, 6]  # curses color constants
  for i, c in enumerate(tile_colors): curses.init_pair(i + 1, c, -1)
  # heat: blue cyan green yellow red
  heat_colors = [curses.COLOR_BLUE, curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED]
  for i, c in enumerate(heat_colors): curses.init_pair(10 + i, c, -1)
  # metrics: white on default
  curses.init_pair(20, curses.COLOR_WHITE, -1)
  # bold green for player
  curses.init_pair(21, curses.COLOR_GREEN, -1)

def heatmap_char(v): return HEAT_CHARS[min(int(np.clip(v, 0, 1) * len(HEAT_CHARS)), len(HEAT_CHARS) - 1)]
def heatmap_cpair(v): return curses.color_pair(10 + min(int(np.clip(v, 0, 1) * 5), 4))

def downsample(arr, th, tw):
  h, w = arr.shape
  bh, bw = max(1, h // th), max(1, w // tw)
  th2, tw2 = min(th, h), min(tw, w)
  return np.array([[arr[r*bh:min((r+1)*bh, h), c*bw:min((c+1)*bw, w)].mean() for c in range(tw2)] for r in range(th2)])

def bar(values, width):
  v = np.clip(np.asarray(values), 0, 1)
  idx = np.minimum((v * len(SPARK)).astype(int), len(SPARK) - 1)
  return ''.join(SPARK[i] for i in idx[:width])

def _safe_addstr(scr, y, x, s, attr=0):
  """addstr that won't crash on edge-of-screen writes"""
  h, w = scr.getmaxyx()
  if y >= h or x >= w: return
  try: scr.addnstr(y, x, s, max(0, w - x), attr)
  except curses.error: pass

def _draw_box(scr, y, x, h, w, label=''):
  _safe_addstr(scr, y, x, '┌─ ' + label + ' ' + '─' * max(0, w - 5 - len(label)) + '┐')
  for r in range(1, h - 1): _safe_addstr(scr, y + r, x, '│'); _safe_addstr(scr, y + r, x + w - 1, '│')
  _safe_addstr(scr, y + h - 1, x, '└' + '─' * (w - 2) + '┘')

def _draw_mid(scr, y, x, w, label=''):
  _safe_addstr(scr, y, x, '├─ ' + label + ' ' + '─' * max(0, w - 5 - len(label)) + '┤')

def render(frame, activations, action, metrics):
  global _scr
  if _scr is None: return
  scr = _scr
  assert frame.shape == (H, W), f"frame shape {frame.shape} != ({H},{W})"

  heat = downsample(np.clip(activations, 0, 1), 12, PANEL_W - 4) if activations is not None else np.zeros((12, PANEL_W - 4))
  act = np.asarray(action) if action is not None else np.zeros(8)
  act_norm = np.clip((act - act.min()) / (act.max() - act.min() + 1e-8), 0, 1)
  conf = float(np.max(act_norm))
  entropy = float(-np.sum(np.clip(act_norm / (act_norm.sum() + 1e-8), 1e-8, 1) * np.log(np.clip(act_norm / (act_norm.sum() + 1e-8), 1e-8, 1))))
  m = metrics or {}

  # world box
  bx_w = W + 2
  _draw_box(scr, 0, 0, H + 2, bx_w, 'WORLD')

  # world tiles
  for r in range(H):
    for c in range(W):
      v = int(frame[r, c])
      ch = GLYPHS.get(v, '?')
      attr = curses.color_pair(v + 1)
      if v == 3: attr |= curses.A_BOLD
      _safe_addstr(scr, r + 1, c + 1, ch, attr)

  # right panel
  px = bx_w
  pw = PANEL_W + 2
  _draw_box(scr, 0, px, H + 2, pw, 'ACTIVATIONS')
  heat_h = heat.shape[0]

  for r in range(heat_h):
    for c in range(heat.shape[1]):
      v = heat[r, c]
      _safe_addstr(scr, r + 1, px + 1 + c, heatmap_char(v), heatmap_cpair(v))

  # player net section
  _draw_mid(scr, heat_h + 1, px, pw, 'PLAYER NET')
  _safe_addstr(scr, heat_h + 2, px + 1, f'action: [{",".join(f"{v:.1f}" for v in act[:8])}]')
  _safe_addstr(scr, heat_h + 3, px + 1, f'confidence: {conf:.2f}')
  _safe_addstr(scr, heat_h + 4, px + 1, bar(act_norm, min(len(act_norm), PANEL_W - 4)))
  _safe_addstr(scr, heat_h + 5, px + 1, f'entropy: {entropy:.2f}')

  # tau display
  _safe_addstr(scr, heat_h + 6, px + 1, f'tau: {m.get("tau", 0):.3f}', curses.color_pair(20))

  # metrics bar
  _draw_mid(scr, H + 1, 0, bx_w, 'METRICS')
  ml = f"tick:{m.get('tick',0)} surprise:{m.get('surprise',0):.2f} coher:{m.get('coherence',0):.2f} pers:{m.get('persistence',0):.2f}"
  mr = f"loss:{m.get('loss',0):.4f} lr:{m.get('lr',0):.0e} params:{m.get('params',0):.1f}K fps:{m.get('fps',0):.1f}"
  _safe_addstr(scr, H + 2, 1, f'{ml}  {mr}', curses.color_pair(20))
  _safe_addstr(scr, H + 3, 0, '└' + '─' * (bx_w - 2) + '┘')

  scr.noutrefresh()
  curses.doupdate()

def init_screen():
  global _scr
  _scr = curses.initscr()
  curses.noecho()
  curses.cbreak()
  curses.curs_set(0)
  _scr.keypad(True)
  _scr.nodelay(True)
  _init_colors()
  _scr.clear()

def cleanup_screen():
  global _scr
  if _scr is None: return
  _scr.keypad(False)
  curses.nocbreak()
  curses.echo()
  curses.curs_set(1)
  curses.endwin()
  _scr = None

if __name__ == '__main__':
  init_screen()
  try:
    frame = np.random.randint(0, N_TILES, (H, W))
    frame[frame > 3] = np.random.choice([0, 1], size=(frame > 3).sum(), p=[0.7, 0.3])
    frame[12, 40] = 3
    for _ in range(8): frame[np.random.randint(H), np.random.randint(W)] = np.random.randint(4, 8)
    acts = np.random.rand(H * 2, W * 2) * np.random.rand(H * 2, W * 2)
    action = np.random.randn(8)
    metrics = dict(tick=1847, surprise=0.34, coherence=0.91, persistence=0.88, loss=0.027, lr=3e-4, params=48.2, fps=12.3, tau=1.15)
    render(frame, acts, action, metrics)
    import time; time.sleep(3)
  finally:
    cleanup_screen()
