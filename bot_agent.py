import torch
import torch.nn as nn
import torch.optim as optim

#this method might be the best way to determine the action
#seperate each action into a different dqn model

"""
# Initialize the DQN models for each query type
state_space_dim = ...  # Size of the state space
#the dimensions would be the parameters of the action space
action_space_dims = {
    "ClaimTerritory": ...,
    "PlaceInitialTroop": ...,
    "RedeemCards": ...,
    "DistributeTroops": ...,
    "Attack": ...,
    "TroopsAfterAttack": ...,
    "Defend": ...,
    "Fortify": ...
}

dqns = {query: DQN(state_space_dim, action_space_dim) for query, action_space_dim in action_space_dims.items()}
optimizers = {query: optim.Adam(dqn.parameters()) for query, dqn in dqns.items()}
loss_fn = nn.MSELoss()

# Training loop for each DQN model
for episode in range(num_episodes):
    state = ...  # Get the initial state
    query = ...  # Determine the current query type
    dqn = dqns[query]
    optimizer = optimizers[query]

    for t in range(max_steps_per_episode):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mask = ...  # Generate the action mask based on the current state

        # Choose an action
        q_values = dqn(state_tensor)
        masked_q_values = q_values * action_mask
        action = torch.argmax(masked_q_values).item()

        next_state, reward, done = ...  # Take the action and get the new state, reward, and done flag

        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])

        # Compute the target value
        with torch.no_grad():
            target_value = reward_tensor + gamma * torch.max(dqn(next_state_tensor)) * (1 - int(done))

        # Update the DQN
        predicted_value = q_values[0, action]
        loss = loss_fn(predicted_value, target_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            break
        state = next_state
"""

# Define the DQN
class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_space_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the DQN
state_space_dim = ...  # Size of the state space
action_space_dim = ...  # Size of the action space
dqn = DQN(state_space_dim, action_space_dim)
optimizer = optim.Adam(dqn.parameters())
loss_fn = nn.MSELoss()

# Train the DQN
for episode in range(num_episodes):
    state = ...  # Get the initial state
    for t in range(max_steps_per_episode):
        action = ...  # Choose an action
        next_state, reward, done = ...  # Take the action and get the new state, reward, and done flag

        # Compute the target Q-value
        target_q_value = reward + gamma * torch.max(dqn(next_state))

        # Update the DQN
        predicted_q_value = dqn(state)[action]
        loss = loss_fn(predicted_q_value, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            break
        state = next_state

def dqn_learn(state, action, reward, next_state, done):
    # Convert states to tensor
    state = torch.FloatTensor(state).unsqueeze(0)
    next_state = torch.FloatTensor(next_state).unsqueeze(0)
    action = torch.LongTensor([action])
    reward = torch.FloatTensor([reward])

    # Predict Q-values for the current state
    predicted_q_values = dqn(state)

    # Compute the target Q-value
    with torch.no_grad():
        next_state_q_values = dqn(next_state)
        max_next_q_value = torch.max(next_state_q_values)
        target_q_value = reward + (gamma * max_next_q_value * (1 - int(done)))

    # Get the Q-value of the action taken
    predicted_q_value = predicted_q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    # Compute loss
    loss = loss_fn(predicted_q_value, target_q_value)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class DQN_Territory_Troops(nn.Module):
    def __init__(self, state_space_dim, num_territories, max_troops):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_space_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_territories * max_troops)  # Output Q-values for each combination

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).view(-1, num_territories, max_troops)  # Reshape to (batch_size, num_territories, max_troops)