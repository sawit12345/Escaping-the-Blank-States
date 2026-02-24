# Escaping-the-Blank-States

Prototype implementation of a universal core-knowledge active inference agent tested on Gymnasium Breakout.

## Idea

This project combines:

- active inference style planning via expected free-energy minimization,
- Spelke-style core systems in one agent:
  - object core: cohesion, continuity, bounded trajectories, permanence,
  - agent core: action-conditioned role inference and goal progress,
  - number core: approximate numerosity belief with surprise tracking,
  - geometry core: boundary evidence and spatial confidence.

The objective is to reduce sample demand and planning compute with priors that are not
hard-coded to a specific game layout.

## Implementation

- `core_priors_ai/perception.py`: generic object-centric perception with background adaptation,
  foreground segmentation, and connected-component object proposals.
- `core_priors_ai/active_inference.py`: full discrete active inference machinery with explicit
  likelihood (`A`), transitions (`B`), preferences (`C`), policy priors (`E`), posterior over
  policies (`q(pi)`), and expected free-energy evaluation.
- `core_priors_ai/agent.py`: unified Spelke cores driving object tracking, role inference,
  numerosity belief updates, geometry confidence, and AIF policy execution.
- `scripts/run_breakout_experiment.py`: reproducible Breakout evaluation script.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run Breakout Test

```bash
python scripts/run_breakout_experiment.py --episodes 3 --max-steps 6000
```

Useful options:

- `--horizon` active inference policy horizon

Per-episode log includes reward, uncertainty, target tracking, and the four core priors
(`obj`, `agent`, `geom`, `num`) plus numerosity statistics, miss count, policy entropy,
state entropy, and mean absolute control offset.

## Notes

This is a research prototype and intentionally avoids deep neural policies. It is designed
to demonstrate how structured priors can support sample-efficient control with lightweight
computation in Atari-like domains.
