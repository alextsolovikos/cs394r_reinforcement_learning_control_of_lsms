# CS394R Final Project - Reinforcement Learning Control of Large Scale Motions in Turbulent Flows

The goal of this project is to design a reinforcement learning (RL) agent that will target fluid volumes (indicated by tracer particles) in a (mean) turbulent boundary layer and move them closer to the wall. Actuation is done with a downwash-inducing jet that accelerates the flow in a localized region of the boundary layer and is controlled by a single input $a_t \in [0,1]$.


This is the code base for the final project of Alex Tsolovikos for CS394R - Reinforcement Learning: Theory and Practice - Spring 2022.

## Requirements
Install the python packages in `requirements.txt`:

```sh
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
$ python -m ipykernel install --user --name=env
`````````

## PPO + LSTM agent with discrete actions
![](figs/ppo_lstm_discrete.gif)
