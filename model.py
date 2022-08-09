import torch
import torch.nn as nn 

class GAN(nn.Module):
    def __init__(self, out_size, latent_size, device=None):
        super().__init__()
        self._latent_size = latent_size
        self._out_size = out_size

        self.generator = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Linear(512,out_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,1),
        )

        self.to(device)

    def to(self, device):
        self.generator.to(device)
        self.critic.to(device)

    def forward_generator(self, noise):
        return self.generator(noise.view(-1,self._latent_size))
    
    def forward_critic(self, x):
        return self.critic(x)