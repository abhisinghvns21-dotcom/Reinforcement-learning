import numpy as np
import random

# States: 0,1,2,3 (2x2 grid)
# Actions: 0=up, 1=down, 2=left, 3=right

Q = np.zeros((4, 4))  # Q-table

alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Reward function
def get_reward(state):
    if state == 3:  # goal state
        return 1
    return 0

# Next state logic
def get_next_state(state, action):
    if action == 3 and state % 2 == 0:   # right
        return state + 1
    elif action == 1 and state < 2:      # down
        return state + 2
    elif action == 2 and state % 2 == 1: # left
        return state - 1
    elif action == 0 and state >= 2:     # up
        return state - 2
    return state  # invalid move -> stay

# Training
for episode in range(1000):
    state = 0
    
    while state != 3:
        
        # Exploration vs Exploitation
        if random.uniform(0,1) < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(Q[state])
        
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        
        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state

print("Q-table:")
print(Q)