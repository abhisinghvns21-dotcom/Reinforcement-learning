import numpy as np
import random

# Grid size
grid_size = 5

# Q-table: (state -> action values)
Q = np.zeros((grid_size, grid_size, 4))  
# 4 actions: up, down, left, right

# Hyperparameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.2    # exploration rate
episodes = 500

# Actions
actions = ['up', 'down', 'left', 'right']

def get_next_state(state, action):
    x, y = state
    
    if action == 0: x = max(x-1, 0)        # up
    elif action == 1: x = min(x+1, grid_size-1)  # down
    elif action == 2: y = max(y-1, 0)      # left
    elif action == 3: y = min(y+1, grid_size-1)  # right
    
    return (x, y)

def get_reward(state):
    if state == (4,4):
        return 10
    return -1

# Training
for ep in range(episodes):
    state = (0,0)
    
    while state != (4,4):
        
        # Exploration vs Exploitation
        if random.uniform(0,1) < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(Q[state[0], state[1]])
        
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        
        # Q-learning update
        old_value = Q[state[0], state[1], action]
        next_max = np.max(Q[next_state[0], next_state[1]])
        
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        Q[state[0], state[1], action] = new_value
        
        state = next_state

print("Training Complete!")

# Map actions to symbols
action_symbols = ['↑', '↓', '←', '→']

print("\nLearned Policy (Best Action at each cell):\n")

for i in range(grid_size):
    row = ""
    for j in range(grid_size):
        if (i, j) == (4,4):
            row += " G  "   # Goal
        else:
            best_action = np.argmax(Q[i, j])
            row += f" {action_symbols[best_action]}  "
    print(row)

print("\nAgent Path from Start to Goal:\n")

state = (0,0)
path = [state]

while state != (4,4):
    action = np.argmax(Q[state[0], state[1]])
    state = get_next_state(state, action)
    path.append(state)

print(path)

grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]

for (x, y) in path:
    grid[x][y] = '*'

grid[4][4] = 'G'

print("\nGrid Path Visualization:\n")
for row in grid:
    print(" ".join(row))