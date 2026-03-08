# Tensor Roguelike Experiments Plan

## Current State
- WorldNet + PlayerNet + interestingness loss all work end-to-end
- CPU only (~5-10s/tick), need GPU
- Loss is stable: surprise ~0.05, coherence ~0.99, persistence ~-0.05
- Early frames are pure noise (untrained weights)
- Only 2 tile types appearing early (mode collapse risk)

## GPU Access (blocking — needed for all experiments)

See experiment 0.

## Experiments

### Exp 0: GPU Access on WSL2
**Goal**: Get tinygrad running on GPU in WSL2
**Approach**: Research in progress — /dev/dxg, CUDA passthrough, CLANG JIT backend
**Success**: <1s/tick instead of ~8s/tick
**Priority**: P0 — blocks everything else

### Exp 1: Loss Function Tuning
**Goal**: Find the sweet spot where structure emerges fastest
**Variables**:
- Surprise vs coherence vs persistence weighting (currently equal, additive)
- Entropy bonus coefficient (currently 0.01)
- Try multiplicative vs additive loss combination
- Temperature annealing schedule (currently linear decay from 2.0 to 0.5)
**Metrics**: Time-to-structure (how many ticks until spatial patterns emerge)
**Protocol**:
1. Run 500 ticks with baseline (current additive loss)
2. Run 500 ticks with multiplicative: -(s * c * p.relu()) with gradient clipping
3. Run 500 ticks with weighted additive: -(2*s + c + 0.5*p)
4. Run 500 ticks with entropy coefficient sweep: 0.001, 0.01, 0.1
5. Compare frame snapshots at tick 50, 100, 200, 500
**Expected**: Multiplicative converges faster but is less stable. Weighting surprise higher should produce more dynamic worlds.

### Exp 2: Architecture Scaling
**Goal**: Find minimum viable model size vs quality
**Variables**:
- HIDDEN dim: 16, 32, 64, 128
- Conv layers: 2, 3, 4
- ACTION_DIM: 8, 16, 32
- Kernel sizes: 3x3, 5x5, 7x7
**Metrics**: params, ticks/sec, loss at tick 500, visual quality score (manual)
**Protocol**:
1. Sweep HIDDEN: [16, 32, 64] × conv layers [2, 3, 4]
2. For each config: run 500 ticks, log loss curve + final frame
3. Plot params vs quality tradeoff
**Expected**: HIDDEN=32 is probably sweet spot on CPU. GPU unlocks 64+.

### Exp 3: Emergence Tracking
**Goal**: Detect and measure when "game-like" properties emerge
**Variables**: None (observational)
**New metrics to implement**:
- **Tile diversity**: entropy over tile type distribution per frame (H = -Σ p_i log p_i)
- **Spatial clustering**: number of connected components of each tile type
- **Entity stability**: do non-floor/wall tiles persist across frames?
- **Symmetry breaking**: does the frame become asymmetric over training?
- **Player influence**: mutual information between action vector and frame delta
**Protocol**:
1. Run 2000 ticks with best loss config from Exp 1
2. Log all new metrics every 10 ticks
3. Plot emergence curves — where do transitions happen?
**Expected**: Phase transitions — sudden jumps in tile diversity, clustering

### Exp 4: Multi-Agent Co-Evolution
**Goal**: Multiple player nets competing/cooperating in same world
**Design**:
- 2-4 PlayerNets, each producing independent action vectors
- WorldNet receives concatenated actions or averaged actions
- Each player has a "position" in the latent space (learned embedding)
- Competition loss: player A's surprise = player B's anti-surprise
**Implementation**:
- Modify main.py to support N players
- Each player gets its own optimizer (or shared with different lr)
- TUI shows each player's attention pattern in different colors
**Expected**: Adversarial dynamics — one player tries to make the world predictable (build walls), the other tries to make it chaotic (blow things up)

### Exp 5: Curriculum Learning — Bootstrap from Simple
**Goal**: Instead of random noise, start from simple patterns and build complexity
**Design**:
- Phase 1 (ticks 0-200): loss = coherence only (learn spatial structure)
- Phase 2 (ticks 200-500): loss = coherence + surprise (learn dynamics)
- Phase 3 (ticks 500+): full interestingness (learn game-like behavior)
- Alternative: start with a small map (20x12), expand to full (80x24)
**Expected**: Much faster convergence. The model learns walls before it learns monsters.

### Exp 6: Reward Shaping — Roguelike Priors
**Goal**: Inject minimal roguelike structure without being prescriptive
**Design**: Additional differentiable loss terms:
- **Room prior**: reward large connected floor regions (convolution-based)
- **Corridor prior**: reward narrow (1-2 wide) floor paths between rooms
- **Entity sparsity**: entities (tiles 3-7) should be <5% of total tiles
- **Player uniqueness**: exactly 1 player tile (@) per frame
- **Wall boundary**: edges should be walls (already done in classic mode)
**Expected**: Dramatically faster emergence of roguelike-looking maps. But less "pure" — we're telling it what a roguelike looks like.

### Exp 7: The Dream — Fully Autoregressive Game
**Goal**: Replace the frame→frame model with a token-level autoregressive model
**Design**:
- Flatten 80x24 frame to 1920 tokens
- Each token = tile type (vocabulary of 8)
- Small transformer: 4 layers, 4 heads, embed_dim=64
- Predict next frame token-by-token, left-to-right, top-to-bottom
- Condition on: previous frame tokens + action embedding
- This IS GPT, but for game frames instead of text
**Why**: Current conv model can only predict the next frame as a whole. Autoregressive model can capture fine-grained spatial dependencies — "if there's a wall at (10,5), there's probably a wall at (10,6)"
**Blocking**: Needs GPU badly. 1920-token sequence with attention is expensive.
**Expected**: Much richer spatial structure. Actual room-like patterns should emerge from the autoregressive factorization.

### Exp 8: Latent Space Visualization
**Goal**: Understand what the nets are "thinking"
**Design**:
- PCA/t-SNE of action vectors over time → do they cluster?
- Decoder probing: feed specific action vectors, see what frames result
- Activation maximization: what input frame maximizes each channel?
- Interpolation: linearly interpolate between two action vectors, render the frames
- Add to TUI: latent space plot (2D projection of recent action vectors)
**Expected**: The action space should develop structure — clusters for "explore", "fight", "build" etc., even without being told these concepts exist.

## Execution Order

```
Exp 0 (GPU) ──────┐
                   ├── Exp 1 (loss tuning) ── Exp 3 (emergence tracking)
                   ├── Exp 2 (arch scaling)
Exp 5 (curriculum) ┘
                        ↓
                   Exp 4 (multi-agent)
                   Exp 6 (priors)
                   Exp 8 (viz)
                        ↓
                   Exp 7 (transformer) ← needs GPU + lessons from 1-6
```

## Infrastructure Needed
- Logging framework: dump metrics to JSON per-tick for analysis
- Snapshot system: save model weights + frame at checkpoints
- Comparison TUI: side-by-side render of two experiments
- Plotting: matplotlib or terminal-based charts for loss curves
