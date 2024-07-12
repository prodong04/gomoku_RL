def compete(agent1, agent2, env):
    state = env.reset()
    done = False
    agents = [agent1, agent2]
    current_agent_idx = 0
    
    while not done:
        current_agent = agents[current_agent_idx]
        action = current_agent.select_action(state)
        state, reward, done, info = env.step(action)
        if "invalid" in info:
            print(f"Invalid move by {current_agent.name}")
            return 3 - env.current_player
        current_agent_idx = 1 - current_agent_idx
        env.render()

    if reward == 1:
        return current_agent.name
    else:
        return "Draw"
