import gym
from simple_dqn_torch import DeepQNetwork, Agent
from utils import plotLearning
import pandas as pd
import numpy as np
from dialogue_manager import Manager

if __name__ == '__main__':
    df = pd.read_csv('data/dataset_clean.csv')[:100]
    df.dropna(inplace=True)
    df.drop(['Weight'], axis=1, inplace=True)
    df = df.groupby('Source')['Target'].apply(list).reset_index()
    dis_sym = {k:v for k,v in df[['Source', 'Target']].values}

    env = Manager(dis_sym)
    n_actions = len(env.symptoms) + len(env.diseases)
    input_dims = len(env.symptoms)
    brain = Agent(gamma=0.99, epsilon=1.0, batch_size=32, n_actions=n_actions,
                  input_dims=[input_dims], alpha=0.003, fc1_dims=100, fc2_dims=100,
                  max_mem_size=100000, eps_end=0.05, eps_dec=0.9996)

    scores = []
    eps_history = []
    num_games = 100000
    score = 0
    # uncomment the line below to record every episode.
    #env = wrappers.Monitor(env, "tmp/space-invaders-1",
    #video_callable=lambda episode_id: True, force=True)
    for i in range(num_games):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score,
                'epsilon %.3f' % brain.EPSILON)
        # else:
            # print('episode: ', i,'score: ', score)
        eps_history.append(brain.EPSILON)
        done = False
        observation = env.reset()
        score = 0
        i = 0
        while not done and i<50:
            action = brain.chooseAction(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            brain.storeTransition(observation, action, reward, observation_,
                                  done)
            observation = observation_
            brain.learn()
            i += 1

        scores.append(score)

    x = [i+1 for i in range(num_games)]
    filename = str(num_games) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
               'Alpha' + str(brain.ALPHA) + 'Memory' + \
                str(brain.Q_eval.fc1_dims) + '-' + str(brain.Q_eval.fc2_dims) +'.png'
    plotLearning(x, scores, eps_history, filename)
