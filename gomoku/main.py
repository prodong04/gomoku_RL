from gomoku_env import GomokuEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from competition import compete

def main():
    env = GomokuEnv(board_size=19)
    
    q_learning_agent1 = QLearningAgent("Q-Learning Agent1", board_size=19)
    q_learning_agent1.learn(episodes=10, env=env)

    q_learning_agent2 = QLearningAgent("Q-Learning Agent2", board_size=19)
    q_learning_agent2.learn(episodes=20, env=env)

    result = compete(q_learning_agent1, q_learning_agent2, env)
    print(f"Winner: {result}")

if __name__ == "__main__":
    main()
