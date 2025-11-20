from simglucose.simulation.scenario_gen import RandomScenario
from datetime import datetime
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from dqn.dqn import DQN
import matplotlib.pyplot as plt
import numpy as np
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.env import T1DSimEnv
from collections import namedtuple, deque
import copy
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.basal_bolus_ctrller import BBController
import random

# Parameters that you can change
hidden_size = 200
gamma = 0.99
tau = 0.01
n_episodes = 1500
replay_memory_size = 1000
batch_size = 128


# You must determine this based on your design
num_obs = ...
num_actions = ...

# Do not change this
Ts = 3 # minutes
steps_per_day = int(8*60/Ts)

#################### Define the Agent ######################
agent = DQN(gamma, tau, hidden_size, num_obs, num_actions, replay_memory_size=replay_memory_size,
            n_episodes=n_episodes,
            batch_size=batch_size)
agent.set_train()


################## Training Loop ##############################
possible_patients = ['adult#00{}'.format(i + 1) for i in range(9)] + ['adult#010']
for episode in range(n_episodes):
    ######## Do not change this section #######
    now = datetime.now()
    patient_name = random.choice(possible_patients)

    # Create a simulation environment
    patient = T1DPatient.withName(patient_name)
    u2ss = patient._params.u2ss
    BW = patient._params.BW
    basal = u2ss * BW / 6000  # unit: U/min

    sensor = CGMSensor.withName('Dexcom', seed=0)
    pump = InsulinPump.withName('Insulet')

    meal_size = np.random.uniform(10, 120)
    scen = [(0.5, meal_size)] # Random Meal size
    scenario = CustomScenario(start_time=now, scenario=scen)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    ################################################

    observation, reward, done, info = env.reset()
    bg = info['bg']
    meal = info['meal']

    ## Define initial state ###
    state = ...

    action_wrapper = namedtuple('action', ['basal', 'bolus'])

    total_time = 0
    t = 0
    done = False
    while not done:

        # We act only when a meal occurs
        if info['meal'] !=0:
            action_idx, bolus = agent.calc_action(np.array(state), epsilon=0)
        else:
            action_idx = None
            bolus = 0

        next_observation, _, terminated, info = env.step(action_wrapper(basal=basal, bolus=bolus))

        done = terminated or t == steps_per_day - 1
        total_time += Ts
        t += 1

        # Define next_state ###
        next_state = ...

        state = next_state

        if done:
            break

        # You must decide when your agent must learn and which are the most important transitions to learn



######################### Plots #############################
# Here (or in other script) you can plot your accum_rewards, accum_losses, and accum_rewards_per_patient


######################### Evaluation #############################

# Similar loop than the training loop, but without learning.
# You can visualize how your model controls the environment with "env.render()"
agent.set_eval()

