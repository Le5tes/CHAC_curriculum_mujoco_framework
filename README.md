# Mujoco RL Framework for CHAC with Curriculum learning

## Credit
Code contained within the subfolder goal_conditioned_baselines is taken from https://github.com/knowledgetechnologyuhh/goal_conditioned_RL_baselines, with some modifications. The main modifications are to be able to run the episodes in parallel using multiprocessing, though I have also made some other small changes to get it to work with my environment and improve performance. The two mujoco xml files in mujoco_sim are also modified from that repo to include the heightmap data. anymal-environment uses the ANYmal model (also modified) from https://github.com/google-deepmind/mujoco_menagerie and the files contained in the assets folder are from there.

## Running
I will provide a docker image with the necessary dependencies installed. The code should be run from within that image.

To train an ant agent, run 
```
python3 train_agent.py <savepath>
```
replacing "<savepath>" with the path you want to save the agent to.

To run an agent, run
```
python3 testrun_agent.py <loadpath>
```
replacing "<loadpath>" with the path to the previously saved agent.

