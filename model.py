import torch
import torch.nn as nn 
from torch.nn.utils import spectral_norm

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
        if device:
            self.to(device)

    def to(self, device):
        self.generator.to(device)
        self.critic.to(device)

    def forward_generator(self, noise):
        return self.generator(noise.view(-1,self._latent_size))
    
    def forward_critic(self, x):
        return self.critic(x)



class SqueezeDimension(nn.Module):
    def forward(self, x):
        return x.squeeze(1)

class AddDimension(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)

def create_generator_architecture(latent_size, out_size):
    return nn.Sequential(nn.Linear(latent_size, 100),
                         nn.LeakyReLU(0.2, inplace=True),
                         AddDimension(),
                         spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
                         nn.Upsample(200),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Upsample(400),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Upsample(800),

                         spectral_norm(nn.Conv1d(32, 1, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),

                         SqueezeDimension(),
                         nn.Linear(800, out_size)
                         )


def create_critic_architecture(data_size):
    return nn.Sequential(AddDimension(),
                         spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.MaxPool1d(2),
                         
                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.MaxPool1d(2),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Flatten(),

                         nn.Linear(224, 50),
                         nn.LeakyReLU(0.2, inplace=True),

                         nn.Linear(50, 15),
                         nn.LeakyReLU(0.2, inplace=True),

                         nn.Linear(15, 1)
                         )


class convGAN(nn.Module):

    def __init__(self, out_size, latent_size, device=None):
        super().__init__()
        self._latent_size = latent_size
        self._out_size = out_size

        self.generator = create_generator_architecture(latent_size, out_size)

        self.critic = create_critic_architecture(out_size)

        if device: 
            self.to(device)

    def to(self, device):
        self.generator.to(device)
        self.critic.to(device)

    def forward_generator(self, noise):
        return self.generator(noise.view(-1,self._latent_size))
    
    def forward_critic(self, x):
        return self.critic(x)

