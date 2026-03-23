import torch
import torch.nn.functional as F

class Diffusion:

    def __init__(self, timesteps=200):

        self.T = timesteps

        self.betas = torch.linspace(1e-4, 0.02, timesteps)

        self.alphas = 1.0 - self.betas

        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        
    def sample_xt(self, x0, t):

        noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].view(-1,1,1,1)

        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        return xt, noise

    def loss(self, model, x0):

        batch_size = x0.shape[0]

        t = torch.randint(0, self.T, (batch_size,))

        xt, noise = self.sample_xt(x0, t)

        pred_noise = model(xt,t)

        loss = F.mse_loss(pred_noise, noise)

        return loss
    def sample(self, model, shape):

        model.eval()

        with torch.no_grad():

            x = torch.randn(shape)

            for t in reversed(range(self.T)):

                t_tensor = torch.full((shape[0],), t, dtype=torch.long)

                alpha = self.alphas[t]
                alpha_bar = self.alpha_bar[t]
                beta = self.betas[t]

                pred_noise = model(x,t_tensor)

                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
                )

                if t > 0:
                    noise = torch.randn_like(x)
                    x += torch.sqrt(beta) * noise

        model.train()

        return x