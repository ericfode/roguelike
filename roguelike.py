#!/usr/bin/env python3
"""the entire game is a tensor graph. there is no game loop, only realize()."""
import sys, os, time, functools, random, struct
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn import Linear

# NOTE: map is a 2d tensor. 0=floor, 1=wall, 2=stairs, 3=player, 4+=monsters
W, H = 80, 24
FLOOR, WALL, STAIRS, PLAYER = 0, 1, 2, 3
GLYPHS = {0: '.', 1: '#', 2: '>', 3: '@', 4: 'g', 5: 'o', 6: 'D', 7: 'T', 8: 'S', 9: 'k'}
COLORS = {0: '90', 1: '37', 2: '33', 3: '32;1', 4: '31', 5: '31;1', 6: '35;1', 7: '36', 8: '33;1', 9: '34'}

# the dungeon generator is a neural net. it THINKS the dungeon into existence
class DungeonBrain:
  def __init__(self, seed=None):
    Tensor.manual_seed(seed or int(time.time()))
    # latent space -> dungeon. the map is a forward pass
    self.l1 = Linear(64, 256)
    self.l2 = Linear(256, 512)
    self.l3 = Linear(512, W*H)
  def dream(self, depth=1):
    z = Tensor.randn(1, 64) * (1 + depth * 0.1)  # deeper = wilder dreams
    x = self.l1(z).relu()
    x = self.l2(x).relu()
    x = self.l3(x)  # raw logits, no sigmoid — let the distribution breathe
    raw = x.reshape(H, W)
    # normalize to 0-1 range per-map, not per-neuron
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    # walls where raw > 0.52 — tuned for ~40% wall density (cave-like)
    tiles = (raw > 0.52).float()
    # cellular automata smoothing: 3x3 average then re-threshold
    kernel = Tensor.ones(1, 1, 3, 3) / 9.0
    for _ in range(2):  # 2 passes of smoothing = organic caves
      smooth = tiles.reshape(1, 1, H, W).pad(((0,0),(0,0),(1,1),(1,1))).conv2d(kernel).reshape(H, W)
      tiles = (smooth > 0.45).float()
    tiles = self._walls(tiles)
    return tiles
  def _walls(self, t):
    # borders are always walls. no escape from the dungeon
    np = t.numpy()
    np[0,:] = 1; np[-1,:] = 1; np[:,0] = 1; np[:,-1] = 1
    return Tensor(np)

# FOV is a convolution. you see by convolving your vision across the map
def fov_kernel(r=6):
  """circular fov kernel. seeing is matrix multiplication."""
  k = 2*r+1
  pts = Tensor.stack(*[Tensor([y-r, x-r]).float() for y in range(k) for x in range(k)]).reshape(k*k, 2)
  dist = (pts * pts).sum(axis=1).sqrt().reshape(k, k)
  return (dist <= r).float().reshape(1, 1, k, k)

def compute_fov(world, px, py, r=6):
  """what you see is what the kernel gives you. fov = conv2d(map, eye_kernel)"""
  k = 2*r+1
  pad = r
  eye = fov_kernel(r)
  # create a point source at player position
  point = Tensor.zeros(H, W)
  np_point = point.numpy(); np_point[py, px] = 1.0
  point = Tensor(np_point).reshape(1, 1, H, W)
  # convolve: spreads the "sight" from player position
  sight = point.pad(((0,0),(0,0),(pad,pad),(pad,pad))).conv2d(eye).reshape(H, W)
  visible = (sight > 0.01).float()
  # walls block sight — crude but fast. multiply by inverse wall density in neighborhood
  wall_density = world.reshape(1, 1, H, W).pad(((0,0),(0,0),(pad,pad),(pad,pad))).conv2d(eye).reshape(H, W) / (k*k)
  # less visible through dense wall areas
  visible = visible * (1.0 - wall_density * 0.7)
  return (visible > 0.3).float()

# combat is dot products. your sword is a vector. the monster is a vector. damage = dot(sword, monster)
class Entity:
  def __init__(self, x, y, kind, hp=10, power=3, defense=1):
    self.pos = Tensor([x, y]).float()
    self.stats = Tensor([hp, hp, power, defense]).float()  # cur_hp, max_hp, power, defense
    self.kind = kind
    self.alive = True
  @property
  def x(self): return int(self.pos[0].item())
  @property
  def y(self): return int(self.pos[1].item())
  @property
  def hp(self): return int(self.stats[0].item())
  @property
  def max_hp(self): return int(self.stats[1].item())
  def attack(self, other):
    """combat is tensor math. damage = relu(my_power - their_defense + noise)"""
    atk = self.stats[2]  # my power
    dfn = other.stats[3]  # their defense
    noise = Tensor.randn(1).item() * 2
    dmg = max(0, int((atk - dfn).item() + noise))
    other.stats = Tensor([max(0, other.hp - dmg), other.stats[1].item(), other.stats[2].item(), other.stats[3].item()]).float()
    if other.hp <= 0: other.alive = False
    return dmg

class World:
  def __init__(self, depth=1, seed=None):
    self.depth, self.turn = depth, 0
    self.brain = DungeonBrain(seed)
    self.base_map = self.brain.dream(depth)  # the terrain tensor
    self.seen = Tensor.zeros(H, W)  # memory of what you've seen
    self.log = []
    # find a floor tile for player
    floors = [(x,y) for y in range(H) for x in range(W) if self.base_map.numpy()[y,x] < 0.5]
    assert len(floors) > 10, "dungeon too dense, bad dream"
    px, py = random.choice(floors[:len(floors)//4])  # start near top-left
    self.player = Entity(px, py, PLAYER, hp=30, power=5, defense=2)
    # spawn monsters on random floor tiles
    self.monsters = []
    monster_floors = [f for f in floors if abs(f[0]-px) + abs(f[1]-py) > 8]
    n_monsters = min(5 + depth * 2, len(monster_floors))
    for i in range(n_monsters):
      mx, my = monster_floors[i]
      kind = random.choice([4, 5, 6, 7, 8, 9])
      hp = 5 + depth * 2 + random.randint(0, 5)
      self.monsters.append(Entity(mx, my, kind, hp=hp, power=2+depth, defense=depth//2))
    # place stairs far from player
    far = max(floors, key=lambda f: abs(f[0]-px) + abs(f[1]-py))
    self.stairs = far
  def msg(self, s): self.log.append(s); self.log = self.log[-5:]

def render(w):
  """rendering is reading the tensor. that's it."""
  vis = compute_fov(w.base_map, w.player.x, w.player.y)
  # update memory
  w.seen = ((w.seen + vis) > 0).float()
  np_map, np_vis, np_seen = w.base_map.numpy(), vis.numpy(), w.seen.numpy()
  out = []
  # entity positions for fast lookup
  ents = {(m.x, m.y): m.kind for m in w.monsters if m.alive}
  ents[(w.stairs[0], w.stairs[1])] = STAIRS
  ents[(w.player.x, w.player.y)] = PLAYER
  for y in range(H):
    row = []
    for x in range(W):
      kind = ents.get((x,y))
      visible = np_vis[y,x] > 0.5
      seen = np_seen[y,x] > 0.5
      if kind is not None and visible:
        g, c = GLYPHS.get(kind, '?'), COLORS.get(kind, '37')
        row.append(f"\033[{c}m{g}\033[0m")
      elif visible:
        tile = int(np_map[y,x] > 0.5)
        g, c = GLYPHS[tile], COLORS[tile]
        row.append(f"\033[{c}m{g}\033[0m")
      elif seen:
        tile = int(np_map[y,x] > 0.5)
        row.append(f"\033[90m{GLYPHS[tile]}\033[0m")
      else:
        row.append(' ')
    out.append(''.join(row))
  # HUD
  hp_bar = f"HP:{w.player.hp}/{w.player.max_hp}"
  depth_str = f"D:{w.depth}"
  turn_str = f"T:{w.turn}"
  monsters_alive = sum(1 for m in w.monsters if m.alive)
  hud = f" {hp_bar} | {depth_str} | {turn_str} | Monsters:{monsters_alive}"
  out.append(f"\033[33;1m{hud}\033[0m")
  # log
  for msg in w.log[-3:]: out.append(f"\033[90m {msg}\033[0m")
  while len(out) < H + 4: out.append('')
  sys.stdout.write('\033[H' + '\n'.join(out[:H+4]) + '\n')
  sys.stdout.flush()

def move_player(w, dx, dy):
  nx, ny = w.player.x + dx, w.player.y + dy
  if nx < 0 or nx >= W or ny < 0 or ny >= H: return
  if w.base_map.numpy()[ny, nx] > 0.5: return  # wall
  # check monster collision = combat
  for m in w.monsters:
    if m.alive and m.x == nx and m.y == ny:
      dmg = w.player.attack(m)
      name = GLYPHS.get(m.kind, '?')
      w.msg(f"you hit {name} for {dmg}. {'killed!' if not m.alive else f'{m.hp}hp left'}")
      return
  # check stairs
  if (nx, ny) == w.stairs:
    w.msg(f"you descend to depth {w.depth+1}...")
    w.__init__(w.depth + 1)
    return
  w.player.pos = Tensor([nx, ny]).float()

def ai_tick(w):
  """monsters chase via tensor distance. pathfinding is just argmin."""
  px, py = w.player.x, w.player.y
  for m in w.monsters:
    if not m.alive: continue
    dist = ((m.pos - w.player.pos) ** 2).sum().sqrt().item()
    if dist > 10: continue  # too far to notice
    # move toward player: pick the axis with larger gap
    dx = 1 if px > m.x else (-1 if px < m.x else 0)
    dy = 1 if py > m.y else (-1 if py < m.y else 0)
    # prefer the larger gap
    if abs(px - m.x) >= abs(py - m.y): dy = 0
    else: dx = 0
    nx, ny = m.x + dx, m.y + dy
    if nx < 0 or nx >= W or ny < 0 or ny >= H: continue
    if w.base_map.numpy()[ny, nx] > 0.5: continue
    # attack player if adjacent
    if nx == px and ny == py:
      dmg = m.attack(w.player)
      w.msg(f"{GLYPHS.get(m.kind,'?')} hits you for {dmg}!")
      if not w.player.alive: w.msg("YOU DIED. the tensors mourn.")
      continue
    # check other monsters blocking
    if any(o.alive and o.x == nx and o.y == ny for o in w.monsters): continue
    m.pos = Tensor([nx, ny]).float()

def get_input():
  """raw terminal input. no curses, no framework. just bytes."""
  import tty, termios
  fd = sys.stdin.fileno()
  old = termios.tcgetattr(fd)
  try:
    tty.setraw(fd)
    ch = sys.stdin.read(1)
    if ch == '\x1b':
      ch += sys.stdin.read(2)
  finally: termios.tcsetattr(fd, termios.TCSADRAIN, old)
  return ch

KEYS = {'h': (-1,0), 'j': (0,1), 'k': (0,-1), 'l': (1,0), 'y': (-1,-1), 'u': (1,-1), 'b': (-1,1), 'n': (1,1),
        '\x1b[A': (0,-1), '\x1b[B': (0,1), '\x1b[C': (1,0), '\x1b[D': (-1,0), '.': (0,0)}

def main():
  w = World(depth=1, seed=42)
  sys.stdout.write('\033[2J\033[?25l')  # clear screen, hide cursor
  try:
    while w.player.alive:
      render(w)
      ch = get_input()
      if ch == 'q': break
      if ch == '?':
        w.msg("hjkl/yubn=move, .=wait, >=stairs, q=quit")
        continue
      if (move := KEYS.get(ch)) is not None:
        if move != (0,0): move_player(w, *move)
        ai_tick(w)
        w.turn += 1
  finally: sys.stdout.write('\033[?25h\033[0m\n')  # restore cursor
  print(f"game over. depth {w.depth}, turn {w.turn}. the tensor graph dissolves.")

if __name__ == '__main__': main()
