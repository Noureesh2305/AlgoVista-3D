import numpy as np
import random

def train_q_learning(alpha=0.1, gamma=0.9, episodes=200):
    grid_size = 4
    num_states = grid_size * grid_size
    num_actions = 4  # Up, Down, Left, Right

    # Rewards: -1 for all, 0 for goal
    rewards = -np.ones((grid_size, grid_size))
    rewards[grid_size - 1][grid_size - 1] = 10  # Goal state reward

    Q = np.zeros((num_states, num_actions))

    def state_to_index(x, y):
        return x * grid_size + y

    def is_valid(x, y):
        return 0 <= x < grid_size and 0 <= y < grid_size

    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    for _ in range(episodes):
        x, y = 0, 0  # Start at top-left
        while (x, y) != (grid_size - 1, grid_size - 1):
            s = state_to_index(x, y)
            if random.uniform(0, 1) < 0.2:
                a = random.randint(0, 3)  # Explore
            else:
                a = np.argmax(Q[s])       # Exploit

            dx, dy = actions[a]
            nx, ny = x + dx, y + dy
            if not is_valid(nx, ny):
                nx, ny = x, y  # Stay if invalid move

            ns = state_to_index(nx, ny)
            reward = rewards[nx][ny]
            Q[s, a] += alpha * (reward + gamma * np.max(Q[ns]) - Q[s, a])
            x, y = nx, ny

    # Extract best path
    path = []
    x, y = 0, 0
    visited = set()
    while (x, y) != (grid_size - 1, grid_size - 1):
        path.append((x, y))
        visited.add((x, y))
        s = state_to_index(x, y)
        a = np.argmax(Q[s])
        dx, dy = actions[a]
        nx, ny = x + dx, y + dy
        if not is_valid(nx, ny) or (nx, ny) in visited:
            break  # Avoid loops
        x, y = nx, ny
    path.append((grid_size - 1, grid_size - 1))  # Append goal

    return Q, grid_size, path
