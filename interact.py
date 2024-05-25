import argparse
import gymnasium as gym
import rldiff_envs


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--env", choices=["CompILE-v0", "NPuzzle-v0"], required=True)
    arg_parser.add_argument("--env_config", default="{}", help="Dictionary as string")
    args = arg_parser.parse_args()

    env = gym.make(f"{args.env}-rldiff", **eval(args.env_config))
    start_state = env.reset()
    print("TOTAL NUMBER OF ACTIONS:", env.action_space.n)
    print(env.unwrapped)
    while True:
        reset = False
        while True:
            try:
                action_str = input('Enter action:')
                action = int(action_str)
                assert 0 <= action < env.action_space.n
                break
            except:
                if action_str.find('q') >= 0:
                    exit()
                if action_str.find('r') >= 0:
                    reset = True
                    break
        if reset:
            start_state = env.reset()
            print(env.unwrapped)
            continue

        next_state, reward, done, _, info = env.step(action)
        print(env.unwrapped)
        if not done:
            print("STATE INDEX:", next_state)
            print("REWARD:", reward)
        else:
            print("DONE!")
            print("STATE INDEX:", next_state)
            print("REWARD:", reward)
            start_state = env.reset()
            print(env.unwrapped)
