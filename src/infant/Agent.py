import torch as T
from torch import nn
from torch import optim
from typing import Tuple

from infant.util import prefer_gpu

class ConvNetModel(nn.Module):
    def __init__(self, output_dims=1024, device=None):
        super(ConvNetModel, self).__init__()
        self.device = device or prefer_gpu()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=2),
            # nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(32, 5, kernel_size=5, stride=2),
            # nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Flatten(),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(16920, output_dims),
            nn.ReLU(),
        ).to(self.device)

    def forward(self, x:T.Tensor):
        x = self.conv_layer(x.to(prefer_gpu()))
        x = self.bottleneck(x)
        return x

class NeuNetModel(nn.Module):
    def __init__(self, lr=1e-5, input_dims=1024, h1_dims=64, h2_dims=64, output_dims=3, device=None):
        super(NeuNetModel, self).__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(input_dims, h1_dims),
            nn.ReLU(),
            nn.Linear(h1_dims, h2_dims),
            nn.ReLU(), 
            nn.Linear(h2_dims, output_dims)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = device if device else prefer_gpu()
        self.to(device)

    def forward(self, x:T.Tensor):
        x = self.model(x)
        return x

class Actor(nn.Module):
    def __init__(self, lr, input_dims=1024, h1_dims=128, h2_dims=128, n_action=3, device=None):
        super(Actor, self).__init__()
        self.lr = lr
        self.base_layer = nn.Sequential(
            nn.Linear(input_dims, h1_dims),
            nn.ReLU(),
            nn.Linear(h1_dims, h2_dims),
            nn.ReLU(), 
        )
        self.mu = nn.Sequential(nn.Linear(h2_dims, n_action))
        self.var = nn.Sequential(nn.Linear(h2_dims, n_action))

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = device if device else prefer_gpu()
        self.to(device)

    def forward(self, x:T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        x = x.to(self.device)
        x = self.base_layer(x)
        mu:T.Tensor = self.mu(x)
        var:T.Tensor = self.var(x)  # log standard deviation
        return mu, var  

class Agent:
    def __init__(self, alpha, beta, gamma=.99, epsilon=.2, max_replay=1_000, n_actions=3):
        self.gamma = gamma
        self.epsilon = epsilon
        self._current_v = None
        self._timestep_ct = 0
        # self._replay_memory = deque(maxlen=max_replay)
        self.n_actions = n_actions
        self.device = prefer_gpu()
        self.log_probs = None

        # self.knowledges = [ConvNetModel()]
        # self.global_knowledge = self.actor
        self.input_layer = ConvNetModel().to(self.device)
        self.actor = Actor(alpha, n_action=self.n_actions, device=self.device)
        self.critic = NeuNetModel(beta, output_dims=1, device=self.device)
    
    def save(self, pathname:str):
        T.save({
            'input_layer': self.input_layer.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_op': self.actor.optimizer.state_dict(),
            'critic_op': self.critic.optimizer.state_dict(),
        }, pathname)

    def load(self, pathname:str):
        checkpoint:dict = T.load(pathname)
        self.input_layer.load_state_dict(checkpoint['input_layer']),
        self.actor.load_state_dict(checkpoint['actor']),
        self.critic.load_state_dict(checkpoint['critic']),
        self.actor.optimizer.load_state_dict(checkpoint['actor_op']),
        self.critic.optimizer.load_state_dict(checkpoint['critic_op']),
        
    def choose_action(self, observation:T.Tensor) -> T.Tensor:
        observation = observation.to(device=self.device)
        x = self.input_layer(observation)

        mus, log_stds = self.actor(x)
        sigmas = T.exp(log_stds)
        action_probs = T.distributions.Normal(mus, sigmas)
        prob_samples = T.squeeze(action_probs.sample((1,)), 2)
        self.log_probs = action_probs.log_prob(prob_samples)
        action = T.squeeze(T.tanh(prob_samples))
            
        return action

    def train(self, state:T.Tensor, reward:float, next_state:T.Tensor, done:bool):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        try:
            x = self.input_layer(state.to(self.device))
            x_ = self.input_layer(next_state.to(self.device))
            critic_value = self.critic(x)
            critic_value_ = self.critic(x_)
        except Exception as e:
            print('Critic: ', e)
            print(state.shape, next_state.shape)
            # print(x, x_)
            return
            
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        delta = reward + self.gamma * critic_value_ * (1-int(done)) - critic_value
        
        actor_loss:T.Tensor = -self.log_probs * delta * self.epsilon
        critic_loss:T.Tensor = delta**2

        grand_loss:T.Tensor = actor_loss + critic_loss
        grand_loss.backward(T.ones_like(grand_loss))

        self.actor.optimizer.step()
        self.critic.optimizer.step()
