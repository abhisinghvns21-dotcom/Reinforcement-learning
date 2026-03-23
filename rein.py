import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# ── Grid & Hyperparameters ───────────────────────────────────────────────────
grid_size = 5
Q = np.zeros((grid_size, grid_size, 4))

alpha   = 0.1   # learning rate
gamma   = 0.9   # discount factor
epsilon = 0.2   # exploration rate
episodes = 500

actions        = ['up', 'down', 'left', 'right']
action_symbols = ['↑',  '↓',   '←',   '→']

# ── Environment helpers ──────────────────────────────────────────────────────
def get_next_state(state, action):
    x, y = state
    if action == 0: x = max(x - 1, 0)
    elif action == 1: x = min(x + 1, grid_size - 1)
    elif action == 2: y = max(y - 1, 0)
    elif action == 3: y = min(y + 1, grid_size - 1)
    return (x, y)

def get_reward(state):
    return 10 if state == (4, 4) else -1

# ── Training ─────────────────────────────────────────────────────────────────
for ep in range(episodes):
    state = (0, 0)
    while state != (4, 4):
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state[0], state[1]])

        next_state = get_next_state(state, action)
        reward     = get_reward(next_state)

        old_value  = Q[state[0], state[1], action]
        next_max   = np.max(Q[next_state[0], next_state[1]])
        Q[state[0], state[1], action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state

print("Training Complete!")

# ── Derive policy & path ─────────────────────────────────────────────────────
policy = np.array([[np.argmax(Q[i, j]) for j in range(grid_size)]
                   for i in range(grid_size)])

state = (0, 0)
path  = [state]
visited = {state}
while state != (4, 4):
    action = np.argmax(Q[state[0], state[1]])
    state  = get_next_state(state, action)
    if state in visited:
        break
    visited.add(state)
    path.append(state)

# Max Q-value heatmap data
max_q = np.max(Q, axis=2)

# ── Arrow offsets for policy arrows ─────────────────────────────────────────
DX = {0: -0.35, 1: 0.35, 2: 0,     3: 0}
DY = {0:  0,    1: 0,    2: -0.35, 3: 0.35}

# ── Custom colormaps ─────────────────────────────────────────────────────────
q_cmap = LinearSegmentedColormap.from_list('q_heat', ['#0d1b2a', '#1a4a6e', '#2196f3', '#64ffda'])

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 6), facecolor='#0d1b2a')
fig.suptitle('Q-Learning · 5×5 Grid World', color='#e0f7fa',
             fontsize=18, fontweight='bold', y=1.02)

gs = GridSpec(1, 3, figure=fig, wspace=0.35)

# ─── Panel 1: Q-value heatmap ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#0d1b2a')
ax1.set_title('Max Q-Value Heatmap', color='#e0f7fa', fontsize=13, pad=10)

im = ax1.imshow(max_q, cmap=q_cmap, aspect='equal')
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color='#90caf9')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#90caf9', fontsize=8)

for i in range(grid_size):
    for j in range(grid_size):
        label = 'G' if (i, j) == (4, 4) else f'{max_q[i, j]:.1f}'
        color = '#ffd700' if (i, j) == (4, 4) else '#e0f7fa'
        ax1.text(j, i, label, ha='center', va='center',
                 fontsize=11 if (i, j) == (4, 4) else 9,
                 fontweight='bold', color=color)

ax1.set_xticks(range(grid_size))
ax1.set_yticks(range(grid_size))
ax1.tick_params(colors='#90caf9')
ax1.set_xlabel('Column', color='#90caf9', fontsize=9)
ax1.set_ylabel('Row',    color='#90caf9', fontsize=9)
for spine in ax1.spines.values():
    spine.set_edgecolor('#1565c0')

# ─── Panel 2: Learned Policy ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#0d1b2a')
ax2.set_title('Learned Policy', color='#e0f7fa', fontsize=13, pad=10)
ax2.set_xlim(-0.5, grid_size - 0.5)
ax2.set_ylim(grid_size - 0.5, -0.5)
ax2.set_aspect('equal')

for i in range(grid_size):
    for j in range(grid_size):
        rect = patches.FancyBboxPatch(
            (j - 0.45, i - 0.45), 0.9, 0.9,
            boxstyle='round,pad=0.04',
            linewidth=1.2,
            edgecolor='#1565c0',
            facecolor='#0a2744'
        )
        ax2.add_patch(rect)

        if (i, j) == (4, 4):
            ax2.text(j, i, 'G', ha='center', va='center', fontsize=14,
                     color='#ffd700', fontweight='bold')
        elif (i, j) == (0, 0):
            ax2.text(j, i, '▶', ha='center', va='center',
                     fontsize=14, color='#64ffda')
        else:
            a = policy[i, j]
            ax2.annotate('', xy=(j + DY[a], i + DX[a]),
                         xytext=(j, i),
                         arrowprops=dict(arrowstyle='->', color='#2196f3', lw=2.0))

ax2.set_xticks(range(grid_size))
ax2.set_yticks(range(grid_size))
ax2.set_xticklabels(range(grid_size), color='#90caf9', fontsize=9)
ax2.set_yticklabels(range(grid_size), color='#90caf9', fontsize=9)
ax2.set_xlabel('Column', color='#90caf9', fontsize=9)
ax2.set_ylabel('Row',    color='#90caf9', fontsize=9)
for spine in ax2.spines.values():
    spine.set_edgecolor('#1565c0')

# ─── Panel 3: Agent Path ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor('#0d1b2a')
ax3.set_title('Agent Path (Start → Goal)', color='#e0f7fa', fontsize=13, pad=10)
ax3.set_xlim(-0.5, grid_size - 0.5)
ax3.set_ylim(grid_size - 0.5, -0.5)
ax3.set_aspect('equal')

for i in range(grid_size):
    for j in range(grid_size):
        rect = patches.FancyBboxPatch(
            (j - 0.45, i - 0.45), 0.9, 0.9,
            boxstyle='round,pad=0.04',
            linewidth=1.2,
            edgecolor='#1565c0',
            facecolor='#0a2744'
        )
        ax3.add_patch(rect)

for step, (i, j) in enumerate(path):
    intensity = 0.3 + 0.7 * step / max(len(path) - 1, 1)
    color = plt.cm.cool(intensity)
    rect = patches.FancyBboxPatch(
        (j - 0.45, i - 0.45), 0.9, 0.9,
        boxstyle='round,pad=0.04',
        linewidth=1.5,
        edgecolor='#64ffda',
        facecolor=(*color[:3], 0.55)
    )
    ax3.add_patch(rect)
    ax3.text(j, i, str(step), ha='center', va='center',
             fontsize=9, color='white', fontweight='bold')

for k in range(len(path) - 1):
    r0, c0 = path[k]
    r1, c1 = path[k + 1]
    ax3.annotate('', xy=(c1, r1), xytext=(c0, r0),
                 arrowprops=dict(arrowstyle='->', color='#ffd700',
                                 lw=1.8, connectionstyle='arc3,rad=0.0'))

ax3.text(path[0][1], path[0][0], 'S', ha='center', va='center',
         fontsize=13, color='#64ffda', fontweight='bold')
ax3.text(4, 4, 'G', ha='center', va='center',
         fontsize=13, color='#ffd700', fontweight='bold')

ax3.set_xticks(range(grid_size))
ax3.set_yticks(range(grid_size))
ax3.set_xticklabels(range(grid_size), color='#90caf9', fontsize=9)
ax3.set_yticklabels(range(grid_size), color='#90caf9', fontsize=9)
ax3.set_xlabel('Column', color='#90caf9', fontsize=9)
ax3.set_ylabel('Row',    color='#90caf9', fontsize=9)
for spine in ax3.spines.values():
    spine.set_edgecolor('#1565c0')

legend_items = [
    patches.Patch(facecolor='#64ffda', edgecolor='#64ffda', label='Start (S)'),
    patches.Patch(facecolor='#ffd700', edgecolor='#ffd700', label='Goal  (G)'),
    patches.Patch(facecolor='#2196f3', edgecolor='#2196f3', alpha=0.6, label='Path steps'),
]
ax3.legend(handles=legend_items, loc='upper right',
           facecolor='#0a2744', edgecolor='#1565c0',
           labelcolor='#e0f7fa', fontsize=8)

plt.tight_layout()
plt.savefig('qlearning_visualization.png', dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
plt.show()
print("Visualization saved → qlearning_visualization.png")