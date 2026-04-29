# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import itertools

# from get_directions import get_directions
# from get_mnist import PointCloudMNIST
# # from get_modelnet import PointCloudModelNet40
# from pipeline import IPTVAEPipeline, compute_loss


# def get_dataset(config: dict):
#     name = config.get("dataset", "mnist")
#     if name == "mnist":
#         return PointCloudMNIST(**config["dataset_kwargs"])
#     # elif name == "modelnet40":
#     #     return PointCloudModelNet40(**config["dataset_kwargs"])
#     else:
#         raise ValueError(f"Unknown dataset: {name}")


# def beta_schedule(epoch: int, warmup_epochs: int, beta_max: float) -> float:
#     """
#     Linear beta warmup: beta rises from 0 to beta_max over warmup_epochs,
#     then stays at beta_max. Starting from 0 prevents the KL term from
#     dominating early training before the Zernike loss has converged.
#     """
#     if warmup_epochs == 0:
#         return beta_max
#     return min(beta_max, beta_max * epoch / warmup_epochs)


# def train(config: dict):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Starting training on device: {device}")

#     directions, weights = get_directions(config["lebedev_order"])
#     directions, weights = directions.to(device), weights.to(device)

#     model = IPTVAEPipeline(
#         directions, weights,
#         l_max=config["l_max"],
#         R=config["R"],
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
#     )

#     print("[INFO] Loading dataset...")
#     data       = get_dataset(config)
#     dataloader = DataLoader(data, batch_size=config["batch_size"], shuffle=True)
#     n          = len(dataloader)

#     print(f"[INFO] Dataset: {config.get('dataset', 'mnist')} | "
#           f"{len(data)} samples | {n} batches per epoch")
#     print(f"[INFO] Beta warmup: 0 → {config['beta_max']} over "
#           f"{config['warmup_epochs']} epochs\n")

#     for epoch in range(config["num_epochs"]):
#         beta = beta_schedule(epoch, config["warmup_epochs"], config["beta_max"])
#         print(f"--- Epoch {epoch+1}/{config['num_epochs']} | beta={beta:.5f} ---")

#         model.train()
#         epoch_losses = {"loss": 0.0, "L_zernike": 0.0, "L_kl": 0.0}

#         for batch_idx, batch in enumerate(dataloader):
#             pc, _ = batch
#             pc = pc.to(device)

#             optimizer.zero_grad()

#             c_pred, c_zernike, mu, logvar_expanded = model(pc)

#             losses = compute_loss(
#                 c_pred          = c_pred,
#                 c_zernike       = c_zernike,
#                 mu              = mu,
#                 logvar_expanded = logvar_expanded,
#                 l_max           = config["l_max"],
#                 beta            = beta,
#             )

#             losses["loss"].backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             for k in epoch_losses:
#                 epoch_losses[k] += losses[k].item()

#             if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n:
#                 print(f"  [Batch {batch_idx+1}/{n}] loss={losses['loss'].item():.4f}")

#         avg = {k: v / n for k, v in epoch_losses.items()}
#         scheduler.step(avg["loss"])

#         print(
#             f">>> END OF EPOCH {epoch+1:03d} SUMMARY |\n"
#             f"    total={avg['loss']:.4f} | "
#             f"zernike={avg['L_zernike']:.4f} | "
#             f"kl={avg['L_kl']:.4f}\n"
#             f"    lr={optimizer.param_groups[0]['lr']} | beta={beta:.5f}\n"
#         )

#     ckpt_path = config.get("checkpoint_path", "checkpoint.pt")
#     torch.save(model.state_dict(), ckpt_path)
#     print(f"[INFO] Saved checkpoint to {ckpt_path}")


# if __name__ == "__main__":
#     # l_max_values   = [2, 4, 6]
#     l_max_values   = [4]
#     lebedev_orders = [50]
#     experiments    = list(itertools.product(l_max_values, lebedev_orders))

#     print(f"[INFO] Queued {len(experiments)} experimental runs.")

#     for l_max, l_order in experiments:
#         print("\n" + "=" * 60)
#         print(f"=== STARTING RUN: l_max = {l_max} | lebedev_order = {l_order} ===")
#         print("=" * 60 + "\n")

#         config = dict(
#             dataset        = "mnist",
#             lebedev_order  = l_order,
#             l_max          = l_max,
#             R              = 8,
#             learning_rate  = 1e-3,
#             num_epochs     = 50,
#             batch_size     = 32,
#             beta_max       = 0.0001,
#             warmup_epochs  = 20,
#             checkpoint_path = f"checkpoint_lmax{l_max}_leb{l_order}.pt",
#             dataset_kwargs = dict(root="./data", train=True, num_points=256),
#         )

#         try:
#             train(config)
#         except Exception as e:
#             print(f"[ERROR] Run failed for l_max={l_max}, lebedev_order={l_order}.")
#             print(f"[ERROR] {e}")
#             continue

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

from get_directions import get_directions
from get_mnist import PointCloudMNIST
# from get_modelnet import PointCloudModelNet40
from pipeline import IPTVAEPipeline, compute_loss


def get_dataset(config: dict):
    name = config.get("dataset", "mnist")
    if name == "mnist":
        return PointCloudMNIST(**config["dataset_kwargs"])
    # elif name == "modelnet40":
    #     return PointCloudModelNet40(**config["dataset_kwargs"])
    else:
        raise ValueError(f"Unknown dataset: {name}")


def beta_schedule(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    """
    Linear beta warmup: beta rises from 0 to beta_max over warmup_epochs,
    then stays at beta_max. Starting from 0 prevents the KL term from
    dominating early training before the Zernike loss has converged.
    """
    if warmup_epochs == 0:
        return beta_max
    return min(beta_max, beta_max * epoch / warmup_epochs)


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
    data       = get_dataset(config)
    dataloader = DataLoader(data, batch_size=config["batch_size"], shuffle=True)
    n          = len(dataloader)

    print(f"[INFO] Dataset: {config.get('dataset', 'mnist')} | "
          f"{len(data)} samples | {n} batches per epoch")
    print(f"[INFO] Beta warmup: 0 → {config['beta_max']} over "
          f"{config['warmup_epochs']} epochs\n")

    for epoch in range(config["num_epochs"]):
        beta = beta_schedule(epoch, config["warmup_epochs"], config["beta_max"])
        print(f"--- Epoch {epoch+1}/{config['num_epochs']} | beta={beta:.5f} ---")

        model.train()
        epoch_losses = {"loss": 0.0, "L_zernike": 0.0, "L_kl": 0.0}

        for batch_idx, batch in enumerate(dataloader):
            pc, _ = batch
            pc = pc.to(device)

            optimizer.zero_grad()

            c_pred, c_zernike, mu, logvar_expanded = model(pc)

            losses = compute_loss(
                c_pred          = c_pred,
                c_zernike       = c_zernike,
                mu              = mu,
                logvar_expanded = logvar_expanded,
                l_max           = config["l_max"],
                beta            = beta,
            )

            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n:
                print(f"  [Batch {batch_idx+1}/{n}] loss={losses['loss'].item():.4f}")

        avg = {k: v / n for k, v in epoch_losses.items()}
        scheduler.step(avg["loss"])

        print(
            f">>> END OF EPOCH {epoch+1:03d} SUMMARY |\n"
            f"    total={avg['loss']:.4f} | "
            f"zernike={avg['L_zernike']:.4f} | "
            f"kl={avg['L_kl']:.4f}\n"
            f"    lr={optimizer.param_groups[0]['lr']} | beta={beta:.5f}\n"
        )

    ckpt_path = config.get("checkpoint_path", "checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    l_max_values   = [2, 4]
    lebedev_orders = [101]
    experiments    = list(itertools.product(l_max_values, lebedev_orders))

    print(f"[INFO] Queued {len(experiments)} experimental runs.")

    for l_max, l_order in experiments:
        print("\n" + "=" * 60)
        print(f"=== STARTING RUN: l_max = {l_max} | lebedev_order = {l_order} ===")
        print("=" * 60 + "\n")

        # Scale beta_max down with l_max to prevent KL domination
        # as the latent space grows with higher l_max.
        beta_max = 0.001 / ((l_max + 1) ** 2)

        config = dict(
            dataset        = "mnist",
            lebedev_order  = l_order,
            l_max          = l_max,
            R              = 8,
            learning_rate  = 1e-3,
            num_epochs     = 50,
            batch_size     = 32,
            beta_max       = beta_max,
            warmup_epochs  = 20,
            checkpoint_path = f"checkpoint_lmax{l_max}_leb{l_order}.pt",
            dataset_kwargs = dict(root="./data", train=True, num_points=256),
        )

        try:
            train(config)
        except Exception as e:
            print(f"[ERROR] Run failed for l_max={l_max}, lebedev_order={l_order}.")
            print(f"[ERROR] {e}")
            continue