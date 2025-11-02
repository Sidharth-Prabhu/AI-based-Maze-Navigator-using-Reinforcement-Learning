#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
import time
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

# -----------------------------
# Maze Environment
# -----------------------------

Action = int
State = Tuple[int, int]

ACTIONS = [0, 1, 2, 3]  # up, right, down, left
DIRS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
ARROWS = {0: '↑', 1: '→', 2: '↓', 3: '←'}


@dataclass
class EnvConfig:
    height: int = 21
    width: int = 21
    start: State = (0, 0)
    goal: State = (-1, -1)  # -1 means bottom-right will be used
    step_penalty: float = -1.0
    wall_penalty: float = -5.0
    goal_reward: float = 50.0
    max_steps_per_episode: Optional[int] = None


class MazeEnv:
    """Grid maze with walls (1) and free cells (0)."""

    def __init__(self, grid: np.ndarray, cfg: EnvConfig):
        assert grid.ndim == 2 and grid.dtype == np.int8
        self.grid = grid
        self.cfg = cfg
        self.h, self.w = grid.shape
        self.start = cfg.start
        self.goal = cfg.goal if cfg.goal != (-1, -
                                             1) else (self.h - 1, self.w - 1)
        for r, c in (self.start, self.goal):
            if not self._in_bounds(r, c) or self.grid[r, c] == 1:
                raise ValueError("Start/goal must be free cells within bounds")
        self.state = self.start
        self.steps = 0
        self.max_steps = cfg.max_steps_per_episode or (self.h * self.w * 4)

    @staticmethod
    def from_generated(cfg: EnvConfig, seed: Optional[int] = None) -> "MazeEnv":
        grid = generate_perfect_maze(cfg.height, cfg.width, seed=seed)
        # Ensure start and goal are empty
        s = cfg.start
        g = cfg.goal if cfg.goal != (-1, -
                                     1) else (cfg.height - 1, cfg.width - 1)
        grid[s[0], s[1]] = 0
        grid[g[0], g[1]] = 0
        return MazeEnv(grid, cfg)

    @staticmethod
    def from_txt(path: str, cfg: EnvConfig) -> "MazeEnv":
        with open(path, 'r') as f:
            rows = [list(line.rstrip('\n')) for line in f]
        h = len(rows)
        w = len(rows[0]) if h else 0
        grid = np.zeros((h, w), dtype=np.int8)
        for i, row in enumerate(rows):
            if len(row) != w:
                raise ValueError("All rows in maze file must have same length")
            for j, ch in enumerate(row):
                grid[i, j] = 1 if ch == '#' else 0
        return MazeEnv(grid, cfg)

    def to_txt(self, path: str) -> None:
        with open(path, 'w') as f:
            for i in range(self.h):
                row = ''.join(
                    '#' if self.grid[i, j] == 1 else '.' for j in range(self.w))
                f.write(row + '\n')

    def reset(self) -> State:
        self.state = self.start
        self.steps = 0
        return self.state

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def step(self, action: Action) -> Tuple[State, float, bool, dict]:
        dr, dc = DIRS[action]
        r, c = self.state
        nr, nc = r + dr, c + dc
        reward = self.cfg.step_penalty
        done = False

        if not self._in_bounds(nr, nc) or self.grid[nr, nc] == 1:
            # hit wall, stay in place
            reward += self.cfg.wall_penalty
            nr, nc = r, c
        else:
            # valid move
            pass

        self.state = (nr, nc)
        self.steps += 1

        if self.state == self.goal:
            reward += self.cfg.goal_reward
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self.state, reward, done, {}

    def render(self, state: Optional[State] = None) -> str:
        s = state if state is not None else self.state
        lines = []
        for i in range(self.h):
            row = []
            for j in range(self.w):
                if (i, j) == s:
                    row.append('A')
                elif (i, j) == self.goal:
                    row.append('G')
                elif (i, j) == self.start:
                    row.append('S')
                else:
                    row.append('#' if self.grid[i, j] == 1 else '.')
            lines.append(''.join(row))
        return '\n'.join(lines)


# -----------------------------
# Maze generation (randomized DFS) -> perfect maze
# -----------------------------

def generate_perfect_maze(height: int, width: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a perfect maze with odd dimensions using randomized DFS on a grid of cells.
    Cells are placed at odd coordinates; walls occupy even coordinates.
    Returns a binary grid: 1 = wall, 0 = free.
    """
    if seed is not None:
        random.seed(seed)
    # Ensure odd dimensions for nice corridors
    if height % 2 == 0:
        height += 1
    if width % 2 == 0:
        width += 1

    grid = np.ones((height, width), dtype=np.int8)

    def neighbors(r, c):
        for dr, dc in [(-2, 0), (0, 2), (2, 0), (0, -2)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1:
                yield nr, nc, r + dr // 2, c + dc // 2  # neighbor cell and wall between

    # Start from a random odd cell
    start_r = random.randrange(1, height, 2)
    start_c = random.randrange(1, width, 2)
    stack = [(start_r, start_c)]
    grid[start_r, start_c] = 0

    while stack:
        r, c = stack[-1]
        nbrs = [(nr, nc, wr, wc)
                for nr, nc, wr, wc in neighbors(r, c) if grid[nr, nc] == 1]
        if not nbrs:
            stack.pop()
            continue
        nr, nc, wr, wc = random.choice(nbrs)
        grid[wr, wc] = 0
        grid[nr, nc] = 0
        stack.append((nr, nc))

    return grid


# -----------------------------
# Q-Learning Agent
# -----------------------------

class QLearningAgent:
    def __init__(self, env: MazeEnv, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.999):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q-table: (H, W, 4)
        self.Q = np.zeros((env.h, env.w, len(ACTIONS)), dtype=np.float32)

    def select_action(self, s: State) -> Action:
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        r, c = s
        return int(np.argmax(self.Q[r, c]))

    def update(self, s: State, a: Action, rwd: float, ns: State, done: bool):
        r, c = s
        nr, nc = ns
        td_target = rwd
        if not done:
            td_target += self.gamma * float(np.max(self.Q[nr, nc]))
        td_error = td_target - self.Q[r, c, a]
        self.Q[r, c, a] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def greedy_action(self, s: State) -> Action:
        r, c = s
        return int(np.argmax(self.Q[r, c]))


# -----------------------------
# Training / Evaluation
# -----------------------------

def train(env: MazeEnv, agent: QLearningAgent, episodes: int, render_every: int = 0,
          verbose: bool = True) -> List[float]:
    returns = []
    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            a = agent.select_action(s)
            ns, r, done, _ = env.step(a)
            agent.update(s, a, r, ns, done)
            s = ns
            ep_return += r
        agent.decay_epsilon()
        returns.append(ep_return)
        if verbose and ep % max(1, episodes // 10) == 0:
            print(
                f"Episode {ep}/{episodes} | Return {ep_return:.1f} | eps={agent.epsilon:.3f}")
        if render_every and ep % render_every == 0:
            print(env.render())
            print()
    return returns


def evaluate(env: MazeEnv, agent: QLearningAgent, episodes: int = 20, render: bool = False) -> Tuple[float, float]:
    total_steps = 0
    successes = 0
    for ep in range(episodes):
        s = env.reset()
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            a = agent.greedy_action(s)
            ns, r, done, _ = env.step(a)
            s = ns
            steps += 1
        total_steps += steps
        if s == env.goal:
            successes += 1
        if render:
            print(
                f"Episode {ep+1}: {'SUCCESS' if s == env.goal else 'FAIL'} in {steps} steps")
            print(env.render())
            print()
    success_rate = successes / episodes
    avg_steps = total_steps / episodes
    return success_rate, avg_steps


def render_policy(env: MazeEnv, agent: QLearningAgent) -> str:
    lines = []
    for i in range(env.h):
        row = []
        for j in range(env.w):
            if env.grid[i, j] == 1:
                row.append('#')
            elif (i, j) == env.goal:
                row.append('G')
            elif (i, j) == env.start:
                row.append('S')
            else:
                a = agent.greedy_action((i, j))
                row.append(ARROWS[a])
        lines.append(''.join(row))
    return '\n'.join(lines)


# -----------------------------
# Persistence helpers
# -----------------------------

def save_q(path: str, Q: np.ndarray) -> None:
    np.save(path, Q)


def load_q(path: str) -> np.ndarray:
    return np.load(path)


# -----------------------------
# CLI
# -----------------------------

def build_env_from_args(args) -> MazeEnv:
    cfg = EnvConfig(height=args.height, width=args.width,
                    start=(args.start_row, args.start_col),
                    goal=(
                        args.goal_row, args.goal_col) if args.goal_row >= 0 and args.goal_col >= 0 else (-1, -1),
                    step_penalty=args.step_penalty,
                    wall_penalty=args.wall_penalty,
                    goal_reward=args.goal_reward,
                    max_steps_per_episode=args.max_steps)
    if args.load_maze:
        env = MazeEnv.from_txt(args.load_maze, cfg)
    else:
        env = MazeEnv.from_generated(cfg, seed=args.seed)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AI-based Maze Navigator using Q-learning")

    sub = parser.add_subparsers(dest='cmd', required=True)

    def add_common(p):
        p.add_argument('--height', type=int, default=21)
        p.add_argument('--width', type=int, default=21)
        p.add_argument('--start-row', type=int, default=0)
        p.add_argument('--start-col', type=int, default=0)
        p.add_argument('--goal-row', type=int, default=-
                       1, help="-1 uses bottom-right")
        p.add_argument('--goal-col', type=int, default=-
                       1, help="-1 uses bottom-right")
        p.add_argument('--step-penalty', type=float, default=-1.0)
        p.add_argument('--wall-penalty', type=float, default=-5.0)
        p.add_argument('--goal-reward', type=float, default=50.0)
        p.add_argument('--max-steps', type=int, default=0,
                       help="0 uses heuristic limit")
        p.add_argument('--seed', type=int, default=None)
        p.add_argument('--load-maze', type=str, default=None)
        p.add_argument('--save-maze', type=str, default=None)

    # train
    p_train = sub.add_parser('train', help='Train a Q-learning agent')
    add_common(p_train)
    p_train.add_argument('--episodes', type=int, default=5000)
    p_train.add_argument('--alpha', type=float, default=0.1)
    p_train.add_argument('--gamma', type=float, default=0.99)
    p_train.add_argument('--epsilon', type=float, default=1.0)
    p_train.add_argument('--epsilon-min', type=float, default=0.05)
    p_train.add_argument('--epsilon-decay', type=float, default=0.999)
    p_train.add_argument('--render-every', type=int, default=0)
    p_train.add_argument('--save', type=str, default=None,
                         help='Path to save Q-table .npy')

    # eval
    p_eval = sub.add_parser('eval', help='Evaluate a trained agent')
    add_common(p_eval)
    p_eval.add_argument('--load', type=str, required=True,
                        help='Path to Q-table .npy')
    p_eval.add_argument('--episodes', type=int, default=50)
    p_eval.add_argument('--render', action='store_true')

    # policy
    p_pol = sub.add_parser('policy', help='Show greedy policy as arrows')
    add_common(p_pol)
    p_pol.add_argument('--load', type=str, required=True)

    # play
    p_play = sub.add_parser('play', help='Play the maze manually with WASD')
    add_common(p_play)

    args = parser.parse_args(argv)

    # Normalize max_steps
    if args.max_steps <= 0:
        args.max_steps = None

    env = build_env_from_args(args)

    if args.save_maze:
        env.to_txt(args.save_maze)
        print(f"Saved maze to {args.save_maze}")

    if args.cmd == 'train':
        agent = QLearningAgent(env, alpha=args.alpha, gamma=args.gamma,
                               epsilon=args.epsilon, epsilon_min=args.epsilon_min,
                               epsilon_decay=args.epsilon_decay)
        returns = train(env, agent, args.episodes,
                        render_every=args.render_every)
        print(
            f"Training complete. Last 10-episode avg return: {np.mean(returns[-10:]):.2f}")
        if args.save:
            save_q(args.save, agent.Q)
            print(f"Saved Q-table to {args.save}")

    elif args.cmd == 'eval':
        Q = load_q(args.load)
        if Q.shape != (env.h, env.w, 4):
            raise ValueError("Loaded Q-table shape does not match environment")
        agent = QLearningAgent(env)
        agent.Q = Q
        agent.epsilon = 0.0
        success, avg_steps = evaluate(
            env, agent, episodes=args.episodes, render=args.render)
        print(f"Success rate: {success*100:.1f}% | Avg steps: {avg_steps:.1f}")

    elif args.cmd == 'policy':
        Q = load_q(args.load)
        if Q.shape != (env.h, env.w, 4):
            raise ValueError("Loaded Q-table shape does not match environment")
        agent = QLearningAgent(env)
        agent.Q = Q
        agent.epsilon = 0.0
        print(render_policy(env, agent))

    elif args.cmd == 'play':
        play(env)


# -----------------------------
# Manual play
# -----------------------------

def play(env: MazeEnv) -> None:
    print("Use WASD to move. q to quit.\n")
    s = env.reset()
    print(env.render())
    while True:
        ch = input("Move (WASD): ").strip().lower()
        if not ch:
            continue
        if ch[0] == 'q':
            break
        key_to_action = {'w': 0, 'd': 1, 's': 2, 'a': 3}
        if ch[0] not in key_to_action:
            print("Invalid key. Use WASD or q.")
            continue
        a = key_to_action[ch[0]]
        ns, r, done, _ = env.step(a)
        print(env.render())
        print(f"Reward: {r}")
        if done:
            if ns == env.goal:
                print("Reached goal!")
            else:
                print("Episode ended (step limit)")
            break


if __name__ == '__main__':
    main()
