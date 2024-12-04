import numpy as np
import random
from collections import defaultdict

# Game Environment
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 3x3 board initialized with 0
        self.done = False  # Game status
        self.winner = None  # Winner: 1 (Agent), -1 (Opponent), 0 (Draw)

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def get_empty_positions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def check_winner(self):
        for i in range(3):
            # Check rows and columns
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                self.done = True
                self.winner = 1 if self.board[i, 0] == 1 else -1
                return
        # Check diagonals
        if abs(self.board.trace()) == 3 or abs(np.fliplr(self.board).trace()) == 3:
            self.done = True
            self.winner = 1 if self.board[1, 1] == 1 else -1
            return
        # Check for draw
        if not self.get_empty_positions():
            self.done = True
            self.winner = 0

    def step(self, action, player):
        if self.done:
            raise ValueError("Game is already over!")
        if self.board[action] != 0:
            raise ValueError("Invalid move!")
        self.board[action] = player
        self.check_winner()
        return self.get_state(), self.winner, self.done

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.2):
        self.q_table = defaultdict(float)  # Q-table
        self.alpha = learning_rate         # Learning rate
        self.gamma = discount_factor       # Discount factor
        self.epsilon = epsilon             # Exploration rate

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)  # Explore
        # Exploit: Choose the best action
        q_values = [self.q_table[(state, action)] for action in actions]
        max_q = max(q_values)
        max_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(max_actions)

    def update_q_value(self, state, action, reward, next_state, done, next_actions):
        old_value = self.q_table[(state, action)]
        if done:
            target = reward
        else:
            max_next_q = max([self.q_table[(next_state, a)] for a in next_actions], default=0)
            target = reward + self.gamma * max_next_q
        self.q_table[(state, action)] = old_value + self.alpha * (target - old_value)

# Train the RL Agent
def train_agent(episodes=5000):
    env = TicTacToe()
    agent = QLearningAgent()
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            actions = env.get_empty_positions()
            action = agent.choose_action(state, actions)
            next_state, winner, done = env.step(action, player=1)

            # Assign rewards
            if done:
                if winner == 1:
                    reward = 1  # Agent wins
                elif winner == -1:
                    reward = -1  # Opponent wins
                else:
                    reward = 0  # Draw
            else:
                # Simulate opponent's random move
                opponent_action = random.choice(env.get_empty_positions())
                next_state, winner, done = env.step(opponent_action, player=-1)
                reward = 0 if not done else -1 if winner == -1 else 0

            next_actions = env.get_empty_positions() if not done else []
            agent.update_q_value(state, action, reward, next_state, done, next_actions)
            state = next_state
    return agent

# Play Against the Agent
def play_against_agent(agent):
    env = TicTacToe()
    state = env.reset()
    print("Your symbol: O (player -1)\n")
    while True:
        print(env.board)
        if env.done:
            if env.winner == 1:
                print("The agent wins!")
            elif env.winner == -1:
                print("You win!")
            else:
                print("It's a draw!")
            break

        # Player's move
        move = tuple(map(int, input("Enter your move (row, col): ").split()))
        try:
            state, winner, done = env.step(move, player=-1)
        except ValueError as e:
            print(e)
            continue

        if done:
            continue

        # Agent's move
        actions = env.get_empty_positions()
        agent_action = agent.choose_action(state, actions)
        state, winner, done = env.step(agent_action, player=1)

# Main
if __name__ == "__main__":
    print("Training the agent...")
    trained_agent = train_agent(episodes=5000)
    print("Agent trained. Let's play!")
    play_against_agent(trained_agent)
