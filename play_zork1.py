import jericho

# Load the Zork1 game
env = jericho.FrotzEnv('games/roms/detective.z5')
obs, info = env.reset()
print(obs)

# Game loop
while True:
    action = input("> ")
    if action == "quit":
        break
    obs, reward, done, info = env.step(action)
    print(obs)
    if done:
        print("Game Over")
        break