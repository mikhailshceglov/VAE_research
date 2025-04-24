import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from vae import VAE
from data import load_cifar10

def train_vae():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _ = load_cifar10(batch_size=32)
    model = VAE(latent_dim=256).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-4)
    recon_vals, kl_vals = [], []

    num_epochs = 20
    kl_anneal = 100          # число эпох для β

    for epoch in range(1, num_epochs+1):
        model.train()
        recon_sum = kl_sum = 0
        beta = min(1.0, epoch/kl_anneal)

        for x, i in tqdm(train_loader, desc=f"Ep {epoch}/{num_epochs}"):
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = VAE.loss_function(recon, x, mu, logvar, beta)
            loss.backward()
            opt.step()

            # для логов отдельно
            recon_sum += F.mse_loss(recon, x, reduction='sum').item()
            kl_sum    += (-0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())).item()

        N = len(train_loader.dataset)
        recon_vals.append(recon_sum/N)
        kl_vals.append(kl_sum/N)
        print(f"Epoch {epoch} | Recon={recon_vals[-1]:.1f} | KL={kl_vals[-1]:.1f} | β={beta:.2f}")

    # сохранить и нарисовать
    torch.save(model.state_dict(), 'vae_model.pt')
    np.save('recon_vals.npy', np.array(recon_vals))
    np.save('kl_vals.npy',   np.array(kl_vals))
    plt.plot(recon_vals, label='Recon MSE')
    plt.plot(kl_vals,    label='KL')
    plt.legend()
    plt.show()

if __name__=='__main__':
    train_vae()
