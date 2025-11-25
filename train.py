from rainbowdqn import Agent as RainbowAgent

agent = RainbowAgent(environment="ALE/Pong-v5")
data = agent.loop(verbose=True)

print(data.loss, data.hns, data.rewards)
