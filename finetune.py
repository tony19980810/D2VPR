# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse
import os
import torch
from lightning.pytorch import callbacks
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback

from finetune_script import my_small_Model
from src.utils import display_datasets_stats
from src.dataloaders.datamodule import VPRDataModule



class SaveModelStateDict(Callback):
    def __init__(self, save_dir="./finetune_weights"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_r1 = -1

    def on_validation_epoch_end(self, trainer, pl_module):

        r1 = trainer.callback_metrics.get("msls-val/R@1")
        if r1 is None:
            return

        r1 = r1.item()


        torch.save({"model_state_dict": pl_module.backbone.state_dict()},
                   os.path.join(self.save_dir, "last_model.pth"))


        if r1 > self.best_r1:
            self.best_r1 = r1
            torch.save({"model_state_dict": pl_module.backbone.state_dict()},
                       os.path.join(self.save_dir, "best_model.pth"))
            print(f"[Info] New best model saved. R@1 = {r1:.4f}")



class HyperParams:
    def __init__(self):
        ## Backbone config:
        self.backbone_name: str = "dinov2_vits14"
        self.unfreeze_n_blocks: int = 2
        
        ## BoQ config:
        self.channel_proj: int = 512
        self.num_queries: int = 64
        self.num_layers: int = 2
        self.output_dim: int = 8192
        
        ## Datasets:
        self.gsv_cities_path: str = "/root/lanyun-tmp/train"
        self.cities: str | list = "all"
        self.val_sets: dict = {
            "msls-val":     "/root/Crica_distill_server/data/val/msls-val",
            "pitts30k-val": "/root/Crica_distill_server/data/val/pitts30k-val",
        }
        
        ## Training config:
        self.batch_size: int = 128
        self.img_per_place: int = 4
        self.max_epochs: int = 120
        self.warmup_epochs: int = 3900
        self.lr: float = 0.0002
        self.weight_decay: float = 0.001
        self.lr_mul: float = 0.1
        self.milestones: list = [10, 20, 30]
        self.num_workers: int = 8
        
        ## misc
        self.silent: bool = False
        self.compile: bool = False
        self.seed: int = 3407



def train(hparams, dev_mode=False):
    seed_everything(hparams.seed, workers=True)
    

    if "dinov2" in hparams.backbone_name:
        train_img_size = (224, 224)
        val_img_size = (224, 224)
        hparams.train_img_size = train_img_size
        hparams.val_img_size = val_img_size

    else:
        raise ValueError(f"backbone {hparams.backbone_name} not implemented!")

    # ---------------- Model ----------------
    model = my_small_Model(
        lr=hparams.lr,
        lr_mul=hparams.lr_mul,
        weight_decay=hparams.weight_decay,
        warmup_epochs=hparams.warmup_epochs,
        milestones=hparams.milestones,
        silent=hparams.silent,
    )

    if hparams.compile:
        model = torch.compile(model)

    # ---------------- Data Module ----------------
    datamodule = VPRDataModule(
        gsv_cities_path=hparams.gsv_cities_path,
        cities=hparams.cities,
        img_per_place=hparams.img_per_place,
        val_sets=hparams.val_sets,
        train_img_size=train_img_size,
        val_img_size=val_img_size,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )

    if not hparams.silent:
        datamodule.setup()
        display_datasets_stats(datamodule)

    # ---------------- TensorBoard ----------------
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs",
        name=f"{hparams.backbone_name}",
        default_hp_metric=False
    )
    tensorboard_logger.log_hyperparams(hparams.__dict__)

    # ---------------- Callbacks ----------------
    callback_list = []

    # Progress bar
    if not hparams.silent:
        callback_list.append(callbacks.RichProgressBar())


    callback_list.append(SaveModelStateDict(save_dir="./finetune_weights"))

    # ---------------- Trainer ----------------
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,
        precision="16-mixed",
        callbacks=callback_list,
        max_epochs=hparams.max_epochs,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        fast_dev_run=dev_mode,
        enable_model_summary=not hparams.silent,
        enable_progress_bar=not hparams.silent,
    )

    trainer.fit(model=model, datamodule=datamodule)



def parse_args():
    parser = argparse.ArgumentParser(description="Train parameters")

    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--seed", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--wd", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--nw", type=int)

    parser.add_argument("--backbone", type=str)
    parser.add_argument("--unfreeze_n", type=int)
    parser.add_argument("--dim", type=int)

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    hparams = HyperParams()
    
    if args.seed:        hparams.seed = args.seed
    if args.compile:     hparams.compile = True
    if args.silent:      hparams.silent = True
    if args.bs:          hparams.batch_size = args.bs
    if args.lr:          hparams.lr = args.lr
    if args.wd:          hparams.weight_decay = args.wd
    if args.epochs:      hparams.max_epochs = args.epochs
    if args.warmup:      hparams.warmup_epochs = args.warmup
    if args.nw:          hparams.num_workers = args.nw
    if args.backbone:    hparams.backbone_name = args.backbone
    if args.unfreeze_n:  hparams.unfreeze_n_blocks = args.unfreeze_n
    if args.dim:         hparams.output_dim = args.dim
    
    train(hparams, dev_mode=args.dev)
