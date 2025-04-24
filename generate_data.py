# generate_data.py
import torch
from vae import VAE
from data import load_cifar10


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VAE(latent_dim=256).to(device)
    model.load_state_dict(torch.load('vae_model.pt', map_location=device))
    model.eval()

    train_loader, _ = load_cifar10(batch_size=128)
    latent_dim = model.latent_dim

    # Считаем суммы mu и logvar для каждого класса
    sum_mu = torch.zeros(10, latent_dim, device=device)
    sum_logvar = torch.zeros(10, latent_dim, device=device)
    counts = torch.zeros(10, device=device)

    for x, labels in train_loader:
        x, labels = x.to(device), labels.to(device)
        with torch.no_grad():
            mu, logvar = model.encode(x)
        for c in range(10):
            mask = (labels == c)
            if mask.any():
                sum_mu[c] += mu[mask].sum(dim=0)
                sum_logvar[c] += logvar[mask].sum(dim=0)
                counts[c] += mask.sum()

    z_means = sum_mu / counts.unsqueeze(1)
    logvar_means = sum_logvar / counts.unsqueeze(1)
    std_means = torch.exp(0.5 * logvar_means)

    # Параметры синтетики
    repeats_per_class = 5000
    batch_size = 128
    sigma = 1.0  # масштаб шума, при необходимости корректировать

    # Генерация синтетических изображений
    all_imgs = []
    all_labels = []
    for c in range(10):
        # формируем латентные коды с шумом
        z_base = z_means[c].unsqueeze(0).repeat(repeats_per_class, 1)
        std_c = std_means[c].unsqueeze(0).repeat(repeats_per_class, 1)
        noise = 1 * std_c * torch.randn_like(z_base)
        z_synth = z_base + noise

        # декодируем пакетами
        for i in range(0, repeats_per_class, batch_size):
            z_batch = z_synth[i:i+batch_size].to(device)
            with torch.no_grad():
                imgs = model.decode(z_batch)
            all_imgs.append(imgs.cpu())
            all_labels.append(torch.full((imgs.size(0),), c, dtype=torch.long))

    # Объединяем и сохраняем
    synthetic_images = torch.cat(all_imgs, dim=0)
    synthetic_labels = torch.cat(all_labels, dim=0)

    torch.save(synthetic_images, 'synthetic_images.pt')
    torch.save(synthetic_labels, 'synthetic_labels.pt')
    print(f"Synthetic dataset generated: {synthetic_images.shape[0]} samples")

if __name__ == '__main__':
    main()