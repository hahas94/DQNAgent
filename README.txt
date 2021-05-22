Authors:
    - Hardy Hasan
    - Marcello Vendruscolo

This is our implementation of DQN:

- In order to train or evaluate the agent on an environment, the testing.py file can be run which has a main.
  Inside the main one needs to choose the environment and whether to train or evaluate.

- In the file train.py, one needs to define paths for saving plots and models. Lines 44 and 152 needs to be changed.

- In the file testing.py, again a path needs to be defined in order to load a saved model. Line 29 needs to be changed.

- In file train.py, two variables need to be defined, namely 'evaluate_freq' and 'evaluation_episodes'. By default
  they are set to 25 and 4 respectively.