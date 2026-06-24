from random import randint

import hydra
import torch
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, ListConfig, OmegaConf


# from lightning.fabric.utilities import measure_flops

OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)

    # FLOPs computation
    # flops = measure_flops(model.cpu(), lambda: model(torch.zeros(1, 3, 256, 256)))
    # print(f"FLOPs: {flops / 10**9}G")

    trainer_config = config.trainer.copy()
    if isinstance(trainer_config.get("callbacks"), DictConfig):
        trainer_config.callbacks = list(trainer_config.callbacks.values())

    logger = hydra.utils.instantiate(config.logger)
    trainer = hydra.utils.instantiate(trainer_config, _target_=Trainer, logger=logger)

    if isinstance(config.mode, ListConfig):
        for mode in config.mode:
            getattr(trainer, mode)(
                model, datamodule=data, ckpt_path=config.checkpoint.get(mode, None)
            )
    else:
        getattr(trainer, config.mode)(
            model, datamodule=data, ckpt_path=config.checkpoint
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
