# visualize_recon.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from data import load_cifar10
from vae import VAE


def denorm(x):
    return x.clamp(0,1)


def plot_training_curves():
    """
    Загружает сохранённые списки recon_vals.npy и kl_vals.npy и рисует кривые обучения
    """
    if not (os.path.exists('recon_vals.npy') and os.path.exists('kl_vals.npy')):
        print('Файлы recon_vals.npy или kl_vals.npy не найдены, пропускаем кривые обучения')
        return
    recon_vals = np.load('recon_vals.npy')
    kl_vals = np.load('kl_vals.npy')
    epochs = np.arange(1, len(recon_vals) + 1)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, recon_vals, label='Recon MSE')
    plt.plot(epochs, kl_vals, label='KL divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss per sample')
    plt.title('VAE Training Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    print('Сохранены кривые обучения в training_curves.png')


def plot_latent_distribution(model, loader, device, max_samples=2000):
    """
    Строит PCA-проекцию µ и σ для первых max_samples образцов из loader
    """
    mus, sigmas, labels = [], [], []
    collected = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            mus.append(mu.cpu())
            sigmas.append(torch.exp(0.5 * logvar).cpu())
            labels.append(y)
            collected += x.size(0)
            if collected >= max_samples:
                break

    mus = torch.cat(mus, dim=0).numpy()[:max_samples]
    sigmas = torch.cat(sigmas, dim=0).numpy()[:max_samples]
    labels = torch.cat(labels, dim=0).numpy()[:max_samples]

    # PCA для µ
    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(mus)
    plt.figure(figsize=(6,6))
    for c in range(10):
        idx = labels == c
        plt.scatter(mu_2d[idx,0], mu_2d[idx,1], s=5, label=str(c))
    plt.title('Latent µ PCA')
    plt.legend(markerscale=2, bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig('latent_mu_pca.png')
    plt.close()
    print('Сохранена PCA µ в latent_mu_pca.png')

    # PCA для σ
    sigma_2d = pca.fit_transform(sigmas)
    plt.figure(figsize=(6,6))
    for c in range(10):
        idx = labels == c
        plt.scatter(sigma_2d[idx,0], sigma_2d[idx,1], s=5, label=str(c))
    plt.title('Latent σ PCA')
    plt.legend(markerscale=2, bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig('latent_sigma_pca.png')
    plt.close()
    print('Сохранена PCA σ в latent_sigma_pca.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Тренировочные кривые
    plot_training_curves()

    # 2) Original vs Reconstruction
    model = VAE(latent_dim=256).to(device)
    model.load_state_dict(torch.load('vae_model.pt', map_location=device))
    model.eval()

    loader, _ = load_cifar10(batch_size=8)
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        recons, _, _ = model(imgs)
    imgs_cpu = imgs.cpu()
    recons_cpu = recons.cpu()

    fig, axes = plt.subplots(2, 8, figsize=(16,4))
    for i in range(8):
        axes[0,i].imshow(denorm(imgs_cpu[i]).permute(1,2,0))
        axes[0,i].axis('off')
        axes[1,i].imshow(denorm(recons_cpu[i]).permute(1,2,0))
        axes[1,i].axis('off')
    axes[0,0].set_ylabel('Original')
    axes[1,0].set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.savefig('reconstructions.png')
    plt.close()
    print('Сохранены реконструкции в reconstructions.png')

    # 3) Распределение латентов
    full_loader, _ = load_cifar10(batch_size=256)
    plot_latent_distribution(model, full_loader, device)
    full_loader, _ = load_cifar10(batch_size=256)
    plot_latent_distribution(model, full_loader, device)

if __name__ == '__main__':
    main()
