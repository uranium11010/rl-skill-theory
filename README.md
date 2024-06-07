# rl-skill-theory

This repository contains code for our ICML 2024 paper,
[When Do Skills Help Reinforcement Learning? A Theoretical Analysis of Temporal Abstractions](https://icml.cc/virtual/2024/poster/35079).
If you use this code, please cite:
```bibtex
@inproceedings{li2022rlskilltheory,
  title={When Do Skills Help Reinforcement Learning? A Theoretical Analysis of Temporal Abstractions},
  author={Li, Zhening and Poesia, Gabriel and Solar-Lezama, Armando},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}
```

## Setup

We use Python 3.9. Run
```
pip install -r requirements.txt
```
to install all dependencies other than PyTorch.
To install PyTorch, follow instructions on its [official webpage](https://pytorch.org/get-started/previous-versions/).

## Experiments determining correlation between RL difficulty and RL sample efficiency ([`train_rl.py`](train_rl.py))

Experiments were conducted on 4 environments:
* `CliffWalking` [1]: a simple grid world (implementation by Gymnasium)
* `CompILE2` [2]: the CompILE grid world with visit length 2
* `8Puzzle`: the 8-puzzle
* `RubiksCube222`: the 2-by-2 Rubiks cube (implementation by `rubiks_cube_gym` [3])

We applied 4 RL algorithms:
* Q-learning
* Value iteration (modified to the RL setting)
* REINFORCE
* DQN

There are 3 sets of experiments:
* Testing how well $p$-learning difficulty and $p$-exploration difficulty
  capture RL sample complexity various macroaction augmentations of the same environment.
  For each environment, we conduct RL on the base environment and 31 macroaction augmentations.
  6 of the 31 macroaction augmentations (located under [`abs_examples/`](abs_examples/))
  are (manually) derived from the optimal macroactions (located under [`abs_optimal/`](abs_optimal/)).
  To reproduce these optimal macroactions with LEMMA [4],
  use the bash scripts located under [`abs_scripts/`](abs_scripts/) for existing configs.
* Testing how well $p$-learning difficulty captures the complexity of planning algorithms
  (state/action value iteration) on various macroaction augmentations of the same environment.
* Testing how well unmerged $p$-incompressibility captures the difficulty in learning
  useful skills for hierarchical RL. In this set of experiments, we test two skill learning algorithms:
  LEMMA [4] for macroactions, and LOVE [5] for neural skills.

Follow the following steps to reproduce our experimental results.
1. First use the bash scripts located under [`envinfo_scripts_main/`](envinfo_scripts_main/)
   to compute information (e.g., RL difficulty metrics) about each environment
   (including both the base environment and the 31 macroaction augmentations).
   The expected amount of time each script takes to run
   varies from a few seconds (`CliffWalking`) to about an hour (`RubiksCube222`).
2. Run the scripts [`scripts_main/QLearning/ENV_true-Q_few-abs-extra[_sN].sh`](scripts_main/QLearning/),
   [`scripts_main/ValueIteration/ENV_true-V_few-abs-extra[_sN].sh`](scripts_main/ValueIteration/),
   and [`scripts_main/REINFORCE/ENV_policy-from-Q_few-abs-extra[_sN].sh`](scripts_main/REINFORCE/),
   for each `ENV` (`CliffWalking`, `CompILE2`, `8Puzzle`, `RubiksCube222`).
   The `_sN` suffix in the script name denotes the seed (absence of the suffix refers to seed 0).
   The expected amount of time each script takes to run
   varies from a few seconds (`CliffWalking`) to a few minutes (`RubiksCube222`).
   These scripts calculate the ground truth state and action values and the optimal policies
   of all 32 variants of each environment.
3. For the experiments with the planning algorithms (state/action value iteration),
   run the scripts [`scripts_main/QLearning/ENV_no-expl_few-abs-extra[_sN].sh`](scripts_main/QLearning/)
   and [`scripts_main/ValueIteration/ENV_no-expl_few-abs-extra[_sN].sh`](scripts_main/ValueIteration/)
   for each `ENV` (`CliffWalking`, `CompILE2`, `8Puzzle`, `RubiksCube222`).
   The `_sN` suffix in the script name denotes the seed (absence of the suffix refers to seed 0).
   The expected amount of time each script takes to run
   varies from several seconds (`CliffWalking`) to a couple tens of minutes (`RubiksCube222`).
   Note that `RubiksCube222` uses the most GPU memory;
   if you're using 8 GPUs at once, then having 15GB of memory available on each is sufficient.
4. Then use the following scripts to conduct all training runs for vanilla RL and hRL with LEMMA macroactions.
    * Q-learning: [`scripts_main/QLearning/ENV_few-abs-extra-trunc-replay-adapteps[_sN].sh`](scripts_main/QLearning/)
    * Value iteration: [`scripts_main/ValueIteration/ENV_few-abs-extra-trunc-replay-adapteps[_sN].sh`](scripts_main/ValueIteration/)
    * REINFORCE: [`scripts_main/REINFORCE/ENV_few-abs-extra-trunc[_sN].sh`](scripts_main/REINFORCE/)
    * DQN: [`scripts_main/QLearning/ENV_deep_few-abs-extra-trunc-replay-adapteps[_sN].sh`](scripts_main/QLearning/)

   The expected amount of time for each script to run varies
   between a couple of hours (`CliffWalking`) to several days (`RubiksCube222`).
   Since there are over a hundred scripts to run,
   we use the script [`track_unfinished.py`](track_unfinished.py) for tracking which runs have completed,
   are in progress, or have not begun.

   We also provide scripts for the following deep RL algorithms that we did not get a chance to experiment on.
   They use the same neural state embedders as DQN:
    * Deep value iteration: [`scripts_main/ValueIteration/ENV_deep_few-abs-extra-trunc-replay-adapteps[_sN].sh`](scripts_main/ValueIteration/) *(not implemented)*
    * Deep REINFORCE: [`scripts_main/REINFORCE/ENV_deep_few-abs-extra-trunc[_sN].sh`](scripts_main/REINFORCE/)
   
   *(Note: We have not yet implemented deep value iteration, so those scripts will not run properly.)*
6. For hRL with LOVE options, first download
   [this zip file](https://drive.google.com/file/d/14hbTFdXMnokwsbZ-NfF7AA0OLYTEIkOd/view?usp=sharing)
   to the root of the repository. It contains LOVE options trained on offline trajectory data
   from each base environment. Extract its contents into `abs_optimal/`:
   ```
   unzip love_ckpts.zip -d abs_optimal
   ```
   Next, download [this zip file](https://drive.google.com/file/d/1tKuBSpq4lu3XiOuTjYXn5I3__XytgnLE/view?usp=sharing)
   to the root of the repository. It contains the offline trajectory data that the LOVE options were trained on.
   Extract its contents:
   ```
   unzip trajectories.zip
   ```
   Finally, run the training scripts
   [`scripts_main/QLearning/ENV_deep_love-trunc-replay-adapteps[_sN].sh`](scripts_main/QLearning/).
   The expected amount of time for each script to run varies
   from a few seconds (`CliffWalking`) to a couple of days (`RubiksCube222`).
   As with the training runs for LEMMA abstractions,
   the script [`track_unfinished.py`](track_unfinished.py) can be used to track progress.
7. To analyze the experimental results, use the Jupyter notebook [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

We have provided the learnt LEMMA macroactions under [`abs_optimal/`](abs_optimal/)
and LOVE options in [this zip file](https://drive.google.com/file/d/14hbTFdXMnokwsbZ-NfF7AA0OLYTEIkOd/view?usp=sharing)
(which you should have downloaded and extracted to [`abs_optimal/`](abs_optimal/) in Step 5 above).
To reproduce all of these abstractions yourself, use the scripts located under [`abs_scripts/`](abs_scripts/).

### References

1. Sutton, R. S. and Barto, A. G. Temporal difference learning.
   In *Reinforcement learning: An introduction*, chapter 6.
   MIT Press, 2018.
2. Kipf, T., Li, Y., Dai, H., Zambaldi, V., Sanchez-Gonzalez,
   A., Grefenstette, E., Kohli, P., and Battaglia, P.
   Compile: Compositional imitation learning and execution.
   In *International Conference on Machine Learning*, pp.
   3418–3428. PMLR, 2019.
3. Hukmani, K., Kolekar, S., and Vobugari, S. Solving twisty
   puzzles using parallel Q-learning. Engineering Letters,
   29(4), 2021.
4. Li, Z., Poesia, G., Costilla-Reyes, O., Goodman, N., and
   Solar-Lezama, A. Lemma: Bootstrapping high-level
   mathematical reasoning with learned symbolic abstractions.
   NeurIPS'22 MATH-AI Workshop, 2022.
5. Jiang, Y., Liu, E., Eysenbach, B., Kolter, J. Z., and Finn, C.
   Learning options via compression. Advances in Neural
   Information Processing Systems, 35:21184–21199, 2022.
