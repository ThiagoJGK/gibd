---
aliases: [gan_pivots_mu_notebook]
tags:
  - grupo/general
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2025-11-27
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Generacion de Pivotes 2025/gan_pivots_mu_notebook.ipynb"
tamanio_bytes: 15998
---

# Notebook: gan_pivots_mu_notebook.ipynb

Ruta interna: `GIBD/Generacion de Pivotes 2025/gan_pivots_mu_notebook.ipynb`

---

# GAN para generar pivotes que maximizan $\Mu$ 
Notebook de ejemplo (PyTorch). 

Incluye:
- Generador (MLP) que produce *m* pivotes (vectores en R^d)
- Critic (DeepSets) estilo WGAN-GP
- Cálculo de $\Mu$ usando selección aleatoria de muestras y de pares (distancia euclidiana)
- Training loop que combina adversarial loss + $-\Mu$ (maximizar Mu)

Nota: el notebook usa embeddings sintéticos por defecto. Cambia la carga de `dataset_embs` por tus embeddings reales.



**[Celda 2 - Código]**
```python
# Imports y utilidades
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from math import sqrt
from itertools import combinations
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

```


**[Celda 3 - Código]**
```python
# Datos de ejemplo: embeddings sintéticos (reemplazá esto con tus embeddings)
# dataset_embs: tensor (N, d)
N = 10000   # número de vectores en la BD (ejemplo)
d = 128     # dimensionalidad de los embeddings
torch.manual_seed(0)
np.random.seed(0)
# Simulamos embeddings con clusters para que el Mu tenga sentido
centroids = np.random.randn(10, d) * 5.0
labels = np.random.randint(0, len(centroids), size=N)
dataset_embs = torch.tensor(centroids[labels] + 0.5*np.random.randn(N, d), dtype=torch.float32).to(device)
# Normalizamos (si tus embeddings están normalizados, conserva la normalización)
dataset_embs = F.normalize(dataset_embs, dim=1)
print('dataset_embs shape:', dataset_embs.shape)

```


**[Celda 4 - Código]**
```python
# Modelos: Generator y Critic (DeepSets-style)
class Generator(nn.Module):
    def __init__(self, noise_dim, m, d, hidden=512):
        super().__init__()
        self.m = m
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, m * d)
        )
    def forward(self, z):
        # z: (batch, noise_dim)
        out = self.net(z)  # (batch, m*d)
        out = out.view(-1, self.m, self.d)  # (batch, m, d)
        # opcional: normalizar cada pivote a la hiperesfera si embeddings normalizados
        out = F.normalize(out, dim=2)
        return out

class CriticDeepSets(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        # per-element embedding
        self.phi = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2)
        )
        # final scoring net on pooled representation
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, X):
        # X: (batch, m, d) or (m,d)
        was_2d = False
        if X.dim() == 2:
            X = X.unsqueeze(0)
            was_2d = True
        # apply phi to each element
        b, m, d = X.shape
        X_flat = X.view(b*m, d)
        H = self.phi(X_flat).view(b, m, -1)
        # mean pooling (DeepSets)
        summary = H.mean(dim=1)  # (b, hidden)
        score = self.rho(summary)  # (b,1)
        if was_2d:
            return score.squeeze(0)
        return score.squeeze(-1)

```


**[Celda 5 - Código]**
```python
# Cálculo de Mu - dos variantes
# Variante A: para cada repetición, muestreamos una submuestra S de tamaño s, calculamos
# la distancia máxima entre pares en S (euclidiana) y promediamos sobre repeticiones.
# Variante B: muestreamos N pares aleatorios y usamos la media de sus distancias.
import math

def pairwise_distances_matrix(X):
    # X: (s,d)
    # returns pairwise distances matrix (s,s)
    with torch.no_grad():
        # usar cdist es directo
        D = torch.cdist(X, X, p=2)
    return D

def mu_variant_A_maxpairs(dataset_embs, num_subsets=64, subset_size=64, device=device):
    # Estimate Mu as average of max pairwise distance within random subsets
    N = dataset_embs.shape[0]
    max_vals = []
    for _ in range(num_subsets):
        idx = torch.randint(0, N, (subset_size,), device=device)
        S = dataset_embs[idx]  # (subset_size, d)
        D = pairwise_distances_matrix(S)  # (s,s)
        # consider only upper triangle excluding diagonal
        iu = torch.triu_indices(subset_size, subset_size, offset=1)
        max_d = D[iu[0], iu[1]].max()
        max_vals.append(max_d.item())
    return float(np.mean(max_vals))

def mu_variant_B_randompairs(dataset_embs, num_pairs=4096, device=device):
    # Estimate Mu as mean distance over num_pairs randomly sampled pairs
    N = dataset_embs.shape[0]
    # sample pairs
    i = torch.randint(0, N, (num_pairs,), device=device)
    j = torch.randint(0, N, (num_pairs,), device=device)
    Xi = dataset_embs[i]
    Xj = dataset_embs[j]
    D = torch.norm(Xi - Xj, dim=1)  # (num_pairs,)
    return float(D.mean().item())

# Quick sanity check on synthetic dataset
print('Mu variant A (max pairs in subsets):', mu_variant_A_maxpairs(dataset_embs, num_subsets=32, subset_size=128))
print('Mu variant B (mean over random pairs):', mu_variant_B_randompairs(dataset_embs, num_pairs=10000))

```


**[Celda 6 - Código]**
```python
# WGAN-GP helpers and regularizers
def gradient_penalty(critic, real, fake, device='cpu', gp_lambda=10.0):
    # real, fake: (batch, m, d)
    batch_size = real.shape[0]
    eps = torch.rand(batch_size, 1, 1, device=device).expand_as(real)
    inter = eps * real + (1 - eps) * fake
    inter.requires_grad_(True)
    score = critic(inter)
    grads = torch.autograd.grad(outputs=score, inputs=inter,
                                grad_outputs=torch.ones_like(score),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gp

def separation_loss(P, sigma=0.1):
    # P: (batch, m, d)
    # penalize collapse of pivots within a set: encourage pairwise distance
    batch, m, d = P.shape
    P_flat = P.view(batch*m, d)
    # compute pairwise distances for each set separately (we'll compute in-batch rough penalty)
    # simple implementation: for each set in batch compute pairwise dists
    loss = 0.0
    for b in range(batch):
        Pb = P[b]  # (m,d)
        D = torch.cdist(Pb, Pb, p=2)  # (m,m)
        # ignore diagonal
        iu = torch.triu_indices(m, m, offset=1)
        if iu.shape[1] > 0:
            # penalize small distances
            vals = torch.exp(- (D[iu[0], iu[1]] ** 2) / (sigma ** 2))
            loss = loss + vals.mean()
    return loss / batch

```


**[Celda 7 - Código]**
```python
# Training loop (sencillo) - WGAN-GP + Mu objective
def train_gan_with_mu(dataset_embs, 
                      m=32, d=128, noise_dim=128,
                      n_iters=2000, batch_size=8, critic_iters=5,
                      lambda_mu=1.0, lambda_gan=1.0, lambda_sep=0.1,
                      mu_mode='A', mu_params=None,
                      device=device):
    # dataset_embs: (N,d)
    G = Generator(noise_dim, m, d).to(device)
    D = CriticDeepSets(d).to(device)
    optG = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    # Precompute dataset size
    N = dataset_embs.shape[0]

    history = {'iter': [], 'mu_est': [], 'd_critic': [], 'lossG': [], 'lossD': []}
    for it in range(n_iters):
        # train critic more times
        for _ in range(critic_iters):
            z = torch.randn(batch_size, noise_dim, device=device)
            fake_sets = G(z)  # (batch, m, d)
            # sample real sets by selecting m real embeddings (to compare set->score)
            idx = torch.randint(0, N, (batch_size, m), device=device)
            real_sets = dataset_embs[idx]  # (batch, m, d)
            # compute critic scores
            real_score = D(real_sets)
            fake_score = D(fake_sets.detach())
            # WGAN loss
            lossD = fake_score.mean() - real_score.mean()
            # gradient penalty
            gp = gradient_penalty(D, real_sets, fake_sets.detach(), device=device)
            lossD_total = lossD + gp
            optD.zero_grad()
            lossD_total.backward()
            optD.step()

        # train generator
        z = torch.randn(batch_size, noise_dim, device=device)
        fake_sets = G(z)
        # adversarial loss (WGAN generator)
        lossG_gan = -D(fake_sets).mean()
        # Mu estimate: for stability we compute Mu on the first generated set in the batch
        # convert to single set (m,d)
        P_single = fake_sets[0].detach().clone()  # note: detach to compute mu from generator? we'll compute mu without detach below
        # compute Mu on dataset using the generator's produced pivots (use the first set)
        # We'll compute Mu using either variant A or B on the DATASET after projecting by pivots if needed
        if mu_mode == 'A':
            # Variant A: for reproducibility use mu_variant_A_maxpairs but applied to dataset as-is
            mu_est = mu_variant_A_maxpairs(dataset_embs, **(mu_params or {'num_subsets':32, 'subset_size':128}), device=device)
        else:
            mu_est = mu_variant_B_randompairs(dataset_embs, **(mu_params or {'num_pairs':4096}), device=device)

        # Note: mu_est is a float; to get a differentiable loss we compute a differentiable
        # surrogate: we compute distances from dataset samples to the generated pivots and use soft-max.
        # For computational cost we sample S points and compute the differentiable version.
        # We'll implement "Mu surrogate" as: average_x( softmax_j dist(x,p_j) ), averaged over S
        S = 256
        idxS = torch.randint(0, N, (S,), device=device)
        S_samples = dataset_embs[idxS]  # (S,d)
        # Distances to pivots (use the first generated set, but keep graph for gradient)
        Dxp = torch.cdist(S_samples, fake_sets[0], p=2)  # (S, m)
        tau = 0.05
        # soft max per x approximating max_j dist(x,p_j)
        dmax_per_x = tau * torch.logsumexp(Dxp / tau, dim=1)  # (S,)
        mu_surrogate = dmax_per_x.mean()
        # Generator separation regularizer to avoid collapse
        loss_sep = separation_loss(fake_sets)
        # Total generator loss: combine adversarial + negative mu_surrogate (we want to maximize Mu)
        lossG_total = lambda_gan * lossG_gan - lambda_mu * mu_surrogate + lambda_sep * loss_sep
        optG.zero_grad()
        lossG_total.backward()
        optG.step()

        # Logging
        if it % 50 == 0 or it == n_iters-1:
            # evaluate Mu on dataset (non-differentiable estimate) using variant A for reporting
            mu_report = mu_variant_A_maxpairs(dataset_embs, num_subsets=32, subset_size=256) if mu_mode=='A' else mu_variant_B_randompairs(dataset_embs, num_pairs=5000)
            d_critic = float(D(fake_sets[0:1].detach()).item())
            history['iter'].append(it)
            history['mu_est'].append(mu_report)
            history['d_critic'].append(d_critic)
            history['lossG'].append(float(lossG_total.item()))
            history['lossD'].append(float(lossD_total.item()))
            print(f'Iter {it:05d} | mu_report {mu_report:.4f} | critic {d_critic:.4f} | lossG {lossG_total.item():.4f} | lossD {lossD_total.item():.4f}')

    return G, D, history

# Quick test run with small iters to validate
G, D, history = train_gan_with_mu(dataset_embs, m=16, d=d, noise_dim=64,
                                 n_iters=200, batch_size=4, critic_iters=3,
                                 lambda_mu=1.0, lambda_gan=1.0, lambda_sep=0.05,
                                 mu_mode='A', mu_params={'num_subsets':16, 'subset_size':64},
                                 device=device)

```


**[Celda 8 - Código]**
```python
# Plotting history (Mu over iterations)
plt.figure(figsize=(6,4))
plt.plot(history['iter'], history['mu_est'], label='Mu estimate (report)')
plt.xlabel('Iteration')
plt.ylabel('Mu (estimate)')
plt.title('Evolución de Mu durante entrenamiento')
plt.legend()
plt.grid(True)
plt.show()

# Show final generated pivots (first set)
P_final = G(torch.randn(1,64, device=device))[0].cpu().numpy()  # (m,d)
print('P_final shape:', P_final.shape)
# plot norms of pivots
norms = np.linalg.norm(P_final, axis=1)
plt.figure(figsize=(6,3))
plt.plot(norms, marker='o')
plt.title('Normas de pivotes generados (primer set)')
plt.xlabel('Pivot index')
plt.ylabel('L2 norm')
plt.grid(True)
plt.show()

```
