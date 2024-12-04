import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque

# Gridworld Environment
class Gridworld:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]  # Starting position
        self.goal_pos = [self.size - 1, self.size - 1]  # Goal position
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        # Flattened one-hot encoding of the agent's position
        state = np.zeros((self.size, self.size))
        state[self.agent_pos[0], self.agent_pos[1]] = 1
        return state.flatten()

    def step(self, action):
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        if action == 0 and self.agent_pos[0] > 0:  # Move up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:  # Move down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # Move left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:  # Move right
            self.agent_pos[1] += 1

        reward = 1 if self.agent_pos == self.goal_pos else -0.1
        done = self.agent_pos == self.goal_pos
        return self._get_state(), reward, done

# Deep Q-Network (DQN)
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

# Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.dqn = DQN(state_size, action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        q_values = self.dqn.model.predict(state)
        return np.argmax(q_values[0])  # Exploit

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.dqn.model.predict(next_state)[0])
            target_f = self.dqn.model.predict(state)
            target_f[0][action] = target
            self.dqn.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training the Agent
def train_agent(episodes=20):
    env = Gridworld(size=5)
    state_size = env.size * env.size
    action_size = 4
    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay()
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    return agent

# Testing the Trained Agent
def test_agent(agent, episodes=5):
    env = Gridworld(size=5)
    state_size = env.size * env.size

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        print(f"Test Episode {e+1}")
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = np.reshape(next_state, [1, state_size])
            print(env.agent_pos)
            if done:
                print("Goal Reached!\n")

# Main
if __name__ == "__main__":
    print("Training the agent...")
    trained_agent = train_agent(episodes=20)
    print("Testing the agent...")
    test_agent(trained_agent)
