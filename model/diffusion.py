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
        t = torch.randint(0, self.T, (batch_size,), device=x0.device)

        data = x0[:, :8]
        coords = x0[:, 8:]

        xt_data, noise = self.sample_xt(data, t)
        xt = torch.cat([xt_data, coords], dim=1)

        pred_noise = model(xt, t)
        pred_noise_data = pred_noise[:, :8]

        eps = 1e-6

        energy_weight = torch.relu(data)

        energy_weight = energy_weight / (energy_weight.mean() + 1e-6)
        energy_weight = torch.clamp(energy_weight, 0.1, 5.0)
        weight = energy_weight

        
        loss_noise = (weight * (pred_noise_data - noise) ** 2).mean()


        loss = loss_noise

        return loss
    def sample(self, model, shape):

        model.eval()

        with torch.no_grad():

            x = torch.randn(shape)
            B, C, H, W = x.shape

            x_coords = torch.linspace(-1, 1, W).view(1,1,1,W).repeat(B,1,H,1)
            y_coords = torch.linspace(-1, 1, H).view(1,1,H,1).repeat(B,1,1,W)

            x_coords = x_coords.to(x.device)
            y_coords = y_coords.to(x.device)
            coords = torch.cat([x_coords, y_coords], dim=1)

            for t in reversed(range(self.T)):

                t_tensor = torch.full((shape[0],), t, dtype=torch.long)
            

                alpha = self.alphas[t].view(1,1,1,1)
                alpha_bar = self.alpha_bar[t].view(1,1,1,1)
                beta = self.betas[t].view(1,1,1,1)

                xt = torch.cat([x, coords], dim=1)
                pred_noise = model(xt, t_tensor)
                pred_noise = pred_noise[:, :8]
                

                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
                )

                if t > 0:
                    noise = torch.randn_like(x)
                    x += torch.sqrt(beta) * noise

        model.train()

        return x