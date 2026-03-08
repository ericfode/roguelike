from tinygrad import Tensor

ACTION_DIM, HIDDEN, H, W = 16, 32, 24, 80

class PlayerNet:
  def __init__(self):
    self.w1, self.b1 = Tensor.kaiming_uniform(HIDDEN, HIDDEN, 4, 4), Tensor.zeros(HIDDEN)
    self.w2, self.b2 = Tensor.kaiming_uniform(HIDDEN, HIDDEN, 3, 3), Tensor.zeros(HIDDEN)
    # conv1: [1,32,24,80] -> [1,32,6,20], conv2: [1,32,6,20] -> [1,32,2,9]
    self.flat_dim = HIDDEN * 2 * 9
    self.wl, self.bl = Tensor.kaiming_uniform(self.flat_dim, ACTION_DIM), Tensor.zeros(ACTION_DIM)

  def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
    assert x.shape == (1, HIDDEN, H, W), f"bad input shape {x.shape}"
    x = x.conv2d(self.w1, self.b1, stride=4).relu()
    x = x.conv2d(self.w2, self.b2, stride=2).relu()
    attn = x.mean(axis=1)  # [1, h, w] spatial attention heatmap
    acts = x.flatten(1) @ self.wl + self.bl  # [1, ACTION_DIM] raw logits
    return acts.reshape(ACTION_DIM), attn.reshape(attn.shape[1], attn.shape[2])

if __name__ == "__main__":
  net = PlayerNet()
  x = Tensor.randn(1, HIDDEN, H, W)
  acts, attn = net(x)
  assert acts.shape == (ACTION_DIM,), f"acts shape {acts.shape}"
  print(f"acts: {acts.shape} attn: {attn.shape} flat_dim: {net.flat_dim}")
  print(f"acts sample: {acts.numpy()[:4]}")
