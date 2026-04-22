import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from get_directions import get_directions
from get_mnist import PointCloudMNIST
from pipeline import IPTVAEPipeline, compute_loss
import itertools


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting training on device: {device}")

    directions, weights = get_directions(config["lebedev_order"])
    directions, weights = directions.to(device), weights.to(device)

    model = IPTVAEPipeline(
        directions, weights,
        l_max=config["l_max"],
        R=config["R"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
    )

    print("[INFO] Loading dataset...")
    data       = PointCloudMNIST(**config["dataset_kwargs"])
    dataloader = DataLoader(data, batch_size=config["batch_size"], shuffle=True)
    n          = len(dataloader)  
    
    print(f"[INFO] Dataset loaded. {len(data)} samples found.")
    print(f"[INFO] Training will run for {config['num_epochs']} epochs, with {n} batches per epoch.\n")

    for epoch in range(config["num_epochs"]):
        print(f"--- Starting Epoch {epoch+1}/{config['num_epochs']} ---")
        model.train()
        epoch_losses = {"L_zernike": 0.0, "L_ipt_sh": 0.0, "L_kl": 0.0, "loss": 0.0}

        for batch_idx, batch in enumerate(dataloader):
            pc, _ = batch
            pc = pc.to(device)

            optimizer.zero_grad()

            c_pred, c_zernike, c_ipt_sh, c_vae_out, mu, logvar = model(pc)

            losses = compute_loss(
                c_pred     = c_pred,
                c_zernike  = c_zernike,
                c_ipt_sh   = c_ipt_sh,
                c_vae_out  = c_vae_out,
                mu         = mu,
                logvar     = logvar,
                l_max      = config["l_max"],
                beta       = config["beta"],
                lambda_ipt = config["lambda_ipt"],
            )

            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
                

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n:
                print(f"  [Epoch {epoch+1} | Batch {batch_idx+1}/{n}] Current Loss: {losses['loss'].item():.4f}")

        avg_loss = epoch_losses["loss"] / n
        scheduler.step(avg_loss)

        print(
            f">>> END OF EPOCH {epoch+1:03d} SUMMARY |\n"
            f"    total={avg_loss:.4f} | "
            f"zernike={epoch_losses['L_zernike']/n:.4f} | "
            f"ipt_sh={epoch_losses['L_ipt_sh']/n:.4f} | "
            f"kl={epoch_losses['L_kl']/n:.4f}\n"
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"    Current Learning Rate: {current_lr}\n")

if __name__ == "__main__":
    l_max_values = [2, 4, 6]
    
    lebedev_orders = [35, 47, 83, 107]
    
    experiments = list(itertools.product(l_max_values, lebedev_orders))
    
    print(f"[INFO] Queued {len(experiments)} experimental runs.")
    
    for l_max, l_order in experiments:
        print("\n" + "="*60)
        print(f"=== STARTING RUN: l_max = {l_max} | lebedev_order = {l_order} ===")
        print("="*60 + "\n")
        
        config = dict(
            lebedev_order  = l_order,     
            l_max          = l_max,
            R              = 8,
            learning_rate  = 1e-3,
            num_epochs     = 10, 
            batch_size     = 32,
            beta           = 0.1,
            lambda_ipt     = 1.0,
            dataset_kwargs = dict(root="./data", train=True, num_points=256),
        )
        
        try:
            train(config)
        except Exception as e:
            print(f"[ERROR] Run failed for l_max={l_max}, lebedev_order={l_order}.")
            print(f"[ERROR] Exception: {e}")
            continue 