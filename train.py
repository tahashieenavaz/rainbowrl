from rainbowrl import Agent

agent = Agent("ALE/Pong-v5")
data = agent.loop(verbose=True)

print(data.loss, data.hns, data.rewards)
