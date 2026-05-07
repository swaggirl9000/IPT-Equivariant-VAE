import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from get_directions import get_directions
from get_mnist import PointCloudMNIST
# from get_shapenet import PointCloudShapeNet
from pipeline import IPTVAEPipeline, compute_loss
from get_modelnet import PointCloudModelNet



def get_dataset(config: dict):
    name = config.get("dataset", "mnist")
    if name == "mnist":
        return PointCloudMNIST(**config["dataset_kwargs"])
    elif name == "shapenet":
        return PointCloudShapeNet(**config["dataset_kwargs"])
    elif name == "shapenet_pc15k":
        from get_shapenet import PointCloudShapeNetPC15k
        return PointCloudShapeNetPC15k(**config["dataset_kwargs"])
    elif name == "modelnet": 
        return PointCloudModelNet(**config["dataset_kwargs"])
    else:
        raise ValueError(f"Unknown dataset: {name}")


def beta_schedule(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    if warmup_epochs == 0:
        return beta_max
    return min(beta_max, beta_max * (epoch + 1) / warmup_epochs)


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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-5
    )

    print("[INFO] Loading dataset...")
    data       = get_dataset(config)
    dataloader = DataLoader(data, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    n          = len(dataloader)

    print(f"[INFO] Dataset: {config.get('dataset')} | "
          f"{len(data)} samples | {n} batches per epoch")
    print(f"[INFO] l_max={config['l_max']} | R={config['R']} | "
          f"lebedev={config['lebedev_order']} | "
          f"beta_max={config['beta_max']:.2e} | warmup={config['warmup_epochs']} epochs\n")

    for epoch in range(config["num_epochs"]):
        beta = beta_schedule(epoch, config["warmup_epochs"], config["beta_max"])

        print(f"--- Epoch {epoch+1}/{config['num_epochs']} | beta={beta:.2e} ---")

        model.train()
        epoch_losses = {"loss": 0.0, "L_zernike": 0.0, "L_ipt_sh": 0.0, "L_kl": 0.0}

        for batch_idx, batch in enumerate(dataloader):
            pc, _ = batch
            pc = pc.to(device)

            optimizer.zero_grad()

            c_pred, c_zernike, c_ipt_sh, c_recon, mu, logvar_expanded = model(pc)

            losses = compute_loss(
                c_pred          = c_pred,
                c_zernike       = c_zernike,
                c_ipt_sh        = c_ipt_sh,
                c_recon         = c_recon,
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
                print(f"  [Batch {batch_idx+1}/{n}] "
                      f"loss={losses['loss'].item():.4f} | "
                      f"zernike={losses['L_zernike'].item():.4f} | "
                      f"ipt_sh={losses['L_ipt_sh'].item():.4f} | "
                      f"kl={losses['L_kl'].item():.1f}")

        avg = {k: v / n for k, v in epoch_losses.items()}
        scheduler.step()

        recon_total = avg["L_zernike"] + avg["L_ipt_sh"]
        print(
            f">>> END OF EPOCH {epoch+1:03d} SUMMARY\n"
            f"    total={avg['loss']:.4f} | "
            f"zernike={avg['L_zernike']:.4f} | "
            f"ipt_sh={avg['L_ipt_sh']:.4f} | "
            f"kl={avg['L_kl']:.2f} | "
            f"beta*kl={beta * avg['L_kl']:.2e}\n"
            f"    lr={optimizer.param_groups[0]['lr']:.2e} | beta={beta:.2e}\n"
        )

        # After epoch 1 (beta=0 probe), print suggested beta_max.
        # if epoch == 0:
            # suggested = recon_total / max(avg["L_kl"], 1.0)
            # print(f"  [HINT] After beta=0 probe: "
            #       f"L_zernike={avg['L_zernike']:.4f}, "
            #       f"L_ipt_sh={avg['L_ipt_sh']:.4f}, "
            #       f"KL={avg['L_kl']:.1f}")
            # print(f"  [HINT] Suggested beta_max ≈ {suggested:.2e}  "
            #       f"(so beta*KL ≈ L_zernike + L_ipt_sh at convergence)\n")

        # Warn if beta is large enough to matter but KL is still unregularised.
        if beta > 0 and avg["L_kl"] > 100:
            print(f"  [WARN] KL={avg['L_kl']:.1f} is still very high at beta={beta:.2e}. "
                  f"beta*KL={beta*avg['L_kl']:.4f} vs recon={recon_total:.4f}.\n"
                  f"  Consider increasing beta_max so beta*KL ~ (L_zernike + L_ipt_sh).\n")

        # Warn if KL term is overwhelming the reconstruction losses.
        if beta > 0 and (beta * avg["L_kl"]) > 10 * recon_total:
            print(f"  [WARN] beta*KL={beta*avg['L_kl']:.4f} >> recon={recon_total:.4f}. "
                  f"The KL term is dominating — reduce beta_max or extend warmup_epochs.\n")

    ckpt_path = config.get("checkpoint_path", "checkpoint.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "l_max":          config["l_max"],
                "R":              config["R"],
                "lebedev_order":  config["lebedev_order"],
            },
        },
        ckpt_path,
    )
    print(f"[INFO] Saved checkpoint to {ckpt_path}")


def load_checkpoint(ckpt_path: str, device):
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict   = raw["state_dict"]
        ckpt_config  = raw.get("config", {})
    else:
        state_dict  = raw
        ckpt_config = {}
        print(f"[WARN] {ckpt_path} is a legacy checkpoint with no metadata. "
              "You must supply l_max, R, lebedev_order manually.")

    return state_dict, ckpt_config

if __name__ == "__main__":
    config = dict(
        dataset        = "modelnet",
        lebedev_order  = 59,
        l_max          = 10,
        R              = 8,
        learning_rate  = 1e-3,
        num_epochs     = 50,
        batch_size     = 32,
        beta_max       = 5e-3,        
        warmup_epochs  = 20,          
        checkpoint_path = "checkpoint_modelnet10_lmax10_R8_leb59_v4.pt",  
        dataset_kwargs = dict(
            root       = "/home/aromanowski/IPT-Equivariant-VAE/data/ModelNet10",
            num_points = 1024,
            split      = "train",
            categories = 10,
            # random_rotate = True,   ← remove this line
        ),
    )
    train(config)

    # config = dict(
    #     dataset        = "shapenet",
    #     lebedev_order  = 29,
    #     l_max          = 10,
    #     R              = 8,
    #     learning_rate  = 1e-3,
    #     num_epochs     = 20,
    #     batch_size     = 32,
    #     beta_max       = 1e-5,    # placeholder — update from [HINT] after epoch 1
    #     warmup_epochs  = 10,
    #     checkpoint_path = "checkpoint_shapenet_lmax10_R8_leb71.pt",
    #     dataset_kwargs = dict(
    #         root       = "./data",
    #         categories = ["Airplane", "Car", "Chair"],
    #         split      = "train",
    #         num_points = 1024,
    #     ),
    # )
    # train(config)
    # config = dict(
    #     dataset         = "mnist",
    #     lebedev_order   = 21,
    #     l_max           = 2,
    #     R               = 8,
    #     learning_rate   = 1e-3,
    #     num_epochs      = 30,
    #     batch_size      = 32,
    #     beta_max        = 0.15,   
    #     warmup_epochs   = 10,
    #     num_workers     = 2,     
    #     checkpoint_path = "checkpoint_mnist_lmax2_R8.pt",
    #     dataset_kwargs  = dict(
    #         root       = "./data",
    #         train      = True,
    #         num_points = 256,
    #     ),
    # )

    # train(config)

    # ----------------------------------------------------------------
    # ShapeNet sweep — uncomment when MNIST trial looks healthy
    # ----------------------------------------------------------------
    # experiments = [
    #     # (l_max, R, lebedev_order)
    #     (32,   8,  71),
    #     (32,  16,  71),
    #     (32,  32,  71),
    #     (64,  16,  101),
    #     (64,  32,  101),
    #     (64,  64,  101),
    #     (128, 32,  131),
    #     (128, 64,  131),
    #     (128, 128, 131),
    # ]
    # for l_max, R, lebedev_order in experiments:
    #     print("\n" + "=" * 60)
    #     print(f"=== RUN: l_max={l_max} | R={R} | lebedev={lebedev_order} ===")
    #     print("=" * 60 + "\n")
    #     config = dict(
    #         dataset        = "shapenet",
    #         lebedev_order  = lebedev_order,
    #         l_max          = l_max,
    #         R              = R,
    #         learning_rate  = 1e-3,
    #         num_epochs     = 50,
    #         batch_size     = 32,
    #         beta_max       = 1e-5,
    #         warmup_epochs  = 10,
    #         checkpoint_path = f"checkpoint_shapenet_lmax{l_max}_R{R}_leb{lebedev_order}.pt",
    #         dataset_kwargs = dict(
    #             root       = "./data",
    #             categories = ["Airplane", "Car", "Chair"],
    #             split      = "train",
    #             num_points = 2048,
    #         ),
    #     )
    #     try:
    #         train(config)
    #     except Exception as e:
    #         print(f"[ERROR] Run failed: l_max={l_max}, R={R}. {e}")
    #         continue
