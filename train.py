import torch
import pytorch_lightning as pl
from configs import get_config
from models import construct_model
from data import construct_data_module
from utils import init_logger, construct_callbacks

if __name__ == "__main__":
    config = get_config()
    pl.seed_everything(config["seed"])
    data_module = construct_data_module(config)
    cem_model = construct_model(config, data_module.imbalance_weight)
    logger = init_logger(config)
    callbacks = construct_callbacks(config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config["ckpt_save_dir"],
        filename="last",  # Lưu với tên 'last.ckpt'
        save_top_k=1,
        monitor=config["ckpt_save_monitor"],
        mode=config["ckpt_save_mode"],
        save_weights_only=True,
    )
    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=config["device"],  
        max_epochs=config["max_epochs"],
        check_val_every_n_epoch=config["val_every_n_epochs"],
        log_every_n_steps=5,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(cem_model, datamodule=data_module)
    trainer.test(cem_model, datamodule=data_module)

    trainer.save_checkpoint(f"{config['ckpt_save_dir']}/last.ckpt")
    print("Mô hình đã được lưu tại", f"{config['ckpt_save_dir']}/last.ckpt")
