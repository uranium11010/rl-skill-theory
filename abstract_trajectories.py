import argparse
import json
import pickle as pkl

from abstractions.abstractions import Axiom
from abstractions.steps import AxStep, Solution
from abstractions.compress import IAPLogN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--action_space_size", type=int, required=True)
    parser.add_argument("--top", type=int)
    args = parser.parse_args()

    with open(args.traj_path, 'rb') as f:
        trajectories = pkl.load(f)
    solutions = [Solution([None] * (len(traj) + 1), [AxStep(action) for _, action in traj]) for traj in trajectories]
    axioms = [Axiom(i) for i in range(args.action_space_size)]
    compressor = IAPLogN(solutions, axioms, config={"abs_type": "ax_seq", "top": args.top})

    abstractions = compressor.abstract()
    abs_list = []
    for ab in abstractions:
        abs_list.append([axiom.name for axiom in ab])

    with open(args.output_path, 'w') as f:
        json.dump(abs_list, f)
