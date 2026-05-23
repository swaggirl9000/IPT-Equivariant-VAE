import torch
recon = torch.load("/home/aromanowski/IPT-Equivariant-VAE/my_pipeline/ipt/results_joint/equivariant_vae_protein/recon_pcs.pt")
gt    = torch.load("/home/aromanowski/IPT-Equivariant-VAE/my_pipeline/ipt/results_joint/equivariant_vae_protein/gt_pcs.pt")
print("recon mean abs:", recon.abs().mean().item())
print("recon std:     ", recon.std().item())
print("gt    mean abs:", gt.abs().mean().item())