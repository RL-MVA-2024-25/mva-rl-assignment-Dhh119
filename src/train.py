from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from hummingbird.ml import convert
import numpy as np
import torch
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import pickle
from tqdm import tqdm
from evaluate import evaluate_HIV, evaluate_HIV_population
from pathlib import Path
import joblib
env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)  
# # The time wrapper limits the number of steps in an episode at 200.
# Now the floor is yours to implement the agent and train it.
# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


class ProjectAgent:
    def __init__(self, regressor=None):
        self.regressor = regressor  # Store the regressor as an attribute

    def act(self, observation, use_random=False):
        actions = env.env.action_set
        candidates = []
        for i in range(4):
            action = actions[i]
            candidates.append(self.regressor.predict(np.concatenate((observation, action)).reshape(1, 8))[0])
        a_index = np.argmax(np.array(candidates))
        return a_index

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.regressor, file)

    def load(self):
        loaded_regressor = joblib.load('regressor1.joblib.xz')
        self.regressor = loaded_regressor


def explore_iter(last_iteration, reg, env=env):
    """
    return sample trajectories for 30 patients after the first iteration : 15% of random exploration
    this function got some changes to try to get better results for the random patients :
    this is the base version that got the final regressor
    """
    # array of dimension (n_samples, 8) (6 state variables, 2 action variables)
    x_samples = []
    reg = convert(reg, 'pytorch')
    for patient in tqdm(range(30), desc='trajectories'):
        # array of dimension (n_samples, 8) (6 state variables, 2 action variables)
        trajectory = []
        state0, _ = env.reset()
        state0 = np.array(state0)
        for epoch in range(200):
            # the choice of the action depends on the last regressor
            u = np.random.rand()
            if u >= 0.85:
                # int between 0 and 3 included : 0 = nothing, 1 = PI, 2 = RTI, 3 = both 4
                a_index = int(np.floor(4*np.random.rand()))
            else:
                candidates_Q = np.array([reg.predict(np.concatenate((state0, action)).reshape(1, 8)) for action in env.env.action_set])
                a_index = np.argmax(candidates_Q)
            action = np.array(env.env.action_set[a_index])
            sample = np.concatenate((state0, action))
            trajectory.append(sample)
            # get the next state
            state0, rew, done, truncated, info = env.step(a_index)
        x_samples.append(trajectory)
    x_samples = np.array(x_samples)
    # concatenate the new training samples with the old ones
    samples = np.load(f'x_samples_{last_iteration}.npy')
    x_samples = np.concatenate((samples, x_samples))
    return x_samples


def explore_random(env=env):
    """
    return sample trajectories for 30 patients
    """
    x_samples = []
    for patient in tqdm(range(30), desc='random trajectories'):
        trajectory = []
        state0, _ = env.reset()
        state0 = np.array(state0)
        for epoch in range(200):
            # choose an action randomly
            # int between 0 and 3 included : 0 = nothing, 1 = PI, 2 = RTI, 3 = both 4
            a_index = int(np.floor(4*np.random.rand()))
            action = np.array(env.env.action_set[a_index])
            sample = np.concatenate((state0, action))
            trajectory.append(sample)  # array of dimension (n_samples, 8) (6 state variables, 2 action variables)
            # get the next state
            state0, rew, done, truncated, info = env.step(a_index)
        x_samples.append(trajectory)  # array of dimension (n_samples, 8) (6 state variables, 2 action variables)
    return np.array(x_samples)


def generate_train(x_samples, reg, env=env):
    """
    generate_train takes as input x_samples (n_samples * 8) and returns the samples using the Q function
    """
    X_train = []
    y_train = []
    max_time = len(x_samples[0])
    nb_patients = len(x_samples)
    model = convert(reg, 'pytorch')
    for j in tqdm(range(nb_patients), desc='patient', leave=False):
        for i in tqdm(range(max_time-1), desc='time', leave=False):
            next_state = x_samples[j][i+1][:6]
            candidates_Q = np.array([model.predict(np.concatenate((next_state, u)).reshape(1, 8)) for u in env.env.action_set])
            Q = np.max(candidates_Q)
            current_state = x_samples[j][i][:6]
            current_action = x_samples[j][i][6:]
            X_train.append(x_samples[j][i])
            y_train.append(env.env.reward(current_state, current_action, next_state) + Q)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train


def Qfit(last_iteration, env=env, use_backup=False):
    """
    Qfit takes as argument an environment, and learn for 400 iterations and approximation of the optimal Q function
    a lot of not useful codes here just for the sake to resuming where I want to 
    """
    finish = False
    while not finish:
        for iteration in tqdm(range(last_iteration+1, 10), desc='iteration'):
            file = Path(f'x_samples_{iteration}.npy')
            if iteration == 0:
                if not file.is_file():
                    x_samples = explore_random(env)
                    np.save(f'x_samples_{iteration}.npy', x_samples)
                else:
                    x_samples = np.load(f'x_samples_{iteration}.npy')
            else:
                if not file.is_file():
                    with open(f'best_regressor_{iteration-1}.pkl', 'rb') as file:
                        reg = pickle.load(file)
                    x_samples = explore_iter(iteration-1, reg, env)
                    np.save(f'x_samples_{iteration}.npy', x_samples)
                else:
                    x_samples = np.load(f'x_samples_{iteration}.npy')
            max_time = len(x_samples[0])
            nb_patients = len(x_samples)
            X_train = []
            y_train = []
            with open('step.txt', 'r') as f:
                last_step = int(f.read())
            for step in tqdm(range(last_step+1, 100), desc='step of the Q-fitted algorithm'):
                if step == 0:
                    for j in tqdm(range(nb_patients), desc='first loop'):
                        for t in range(max_time-1):
                            next_state = x_samples[j][t+1][:6]
                            candidates_Q = np.array([env.env.reward(next_state, u, next_state) for u in env.env.action_set])
                            Q = np.max(candidates_Q)
                            current_state = x_samples[j][t][:6].copy()
                            current_action = x_samples[j][t][6:].copy()
                            X_train.append(x_samples[j][t].copy())
                            y_train.append(env.env.reward(current_state, current_action, next_state)+Q)
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    reg = ExtraTreesRegressor(n_estimators=50, min_samples_split=2).fit(X_train, y_train)
                    with open(f'regressor_{iteration}.pkl', 'wb') as file:
                        pickle.dump(reg, file)
                    with open(file='step.txt', mode='w') as f:
                        f.write(f'{step}')
                else:
                    # in case disconnect when writing in the main file
                    # better to try except, but I forgot the error that it raised
                    if use_backup:
                        with open('backup.pkl', 'rb') as file:
                            reg = pickle.load(file)
                    else:
                        with open(f'regressor_{iteration}.pkl', 'rb') as file:
                            reg = pickle.load(file)
                    X_train, y_train = generate_train(x_samples, reg, env=env)
                    reg = ExtraTreesRegressor(n_estimators=50, min_samples_split=2).fit(X_train, y_train)
                    with open(f'regressor_{iteration}.pkl', "wb") as file:
                        pickle.dump(reg, file)
                    with open(file='step.txt', mode='w') as f:
                        f.write(f'{step}')
                    with open('backup.pkl', 'wb') as file:
                        pickle.dump(reg, file)
                agent = ProjectAgent(regressor=reg)
                score = evaluate_HIV(agent, nb_episode=1)
                file = Path('best_score.txt')
                if not file.is_file():
                    with open('best_score.txt', 'w') as f:
                        f.write(f'{score}')
                with open('best_score.txt', 'r') as f:
                    best_score = float(f.read())
                print('score', score, 'best_score', best_score)
                if best_score < score:
                    with open('best_score.txt', 'w') as f:
                        f.write(f'{score}')
                    with open(f'best_regressor_{iteration}.pkl', 'wb') as file:
                        pickle.dump(reg, file)
                if score > 5e10:
                    with open('best_regressor.pkl', 'wb') as file:
                        pickle.dump(reg, file)
                    finish = True
            with open(file='step.txt', mode='w') as f:
                f.write('-1')
    with open('fineshed.txt', 'w') as f:
        f.write('done')

# strategy : inspired by paper
# random action for 200 epochs (exploration) then Qfit
# iter 10 times : random exploration (15%), use Q found in last iteration (85%) for 200 epochs then Qfit
# this adds a lot of training samples
# finally Qfit on all those samples

# to keep in mind : modify the Q function because in the grading
# the evaluation that matters is not the traditional sum over gamma^t r_t
# but sum over 200 epochs of r_t
# to overcome this issue : replace the function to fit by c(x, u) + max(hat(Q)_n(x', u))


