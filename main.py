import random
import math
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from itertools import count
from replay_memory import ReplayMemory, Transition
from dqn_model import DQN
from skip_wrapper import MaxAndSkipEnv
from utils import get_screen


WINDOW_LENGTH = 4
SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUM_EPISODES = 50
MEMORY_SIZE = 1000000

steps_done = 0
writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = 'SpaceInvaders-v0'
env = gym.make(env_name)
env = MaxAndSkipEnv(env, skip=8)
env = gym.wrappers.Monitor(env, './videos/' + env_name, force=True, mode='training')

np.random.seed(123)
env.seed(123)

NUM_ACTIONS = env.action_space.n

policy_net = DQN(SCREEN_HEIGHT, SCREEN_WIDTH, NUM_ACTIONS).to(device)
target_net = DQN(SCREEN_HEIGHT, SCREEN_WIDTH, NUM_ACTIONS).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(MEMORY_SIZE)

def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(NUM_ACTIONS)]], device=device, dtype=torch.long)

def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	# (a final state would've been the one after which simulation ended)
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
											batch.next_state)), device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state
												if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	state_action_values = policy_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for non_final_next_states are computed based
	# on the "older" target_net; selecting their best reward with max(1)[0].
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	criterion = nn.SmoothL1Loss()
	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	writer.add_scalar('Loss', loss, steps_done)

	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()



if __name__ == '__main__':

	for i_episode in range(NUM_EPISODES):
		# Initialize the environment and state
		env.reset()
		last_screen = get_screen(env)
		current_screen = get_screen(env)
		state = current_screen - last_screen
		total_reward = 0
		for t in count():
			steps_done += 1
			# Select and perform an action
			action = select_action(state, steps_done)
			_, reward, done, _ = env.step(action.item())
			reward = torch.tensor([reward], device=device)
			total_reward += reward
			# Observe new state
			last_screen = current_screen
			current_screen = get_screen(env)
			if not done:
				next_state = current_screen - last_screen
			else:
				next_state = None

			# Store the transition in memory
			memory.push(state, action, next_state, reward)

			# Move to the next state
			state = next_state

			# Perform one step of the optimization (on the policy network)
			optimize_model()

			if done:
				print("Episode {} finished after {} timesteps - Reward: {}".format(i_episode+1, t + 1, total_reward.item()))
				writer.add_scalar('Reward', total_reward.item(), i_episode)
				break

		# Update the target network, copying all weights and biases in DQN
		if i_episode > 0 and i_episode % TARGET_UPDATE == 0:
			print("Updating target network")
			target_net.load_state_dict(policy_net.state_dict())
			torch.save(policy_net.state_dict(), './models/' + env_name + '_' + str(i_episode) + '.pth')

	env.close()