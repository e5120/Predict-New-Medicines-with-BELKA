import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from lb import LBDataModule, LBModelModule
from lb.utils import setup, get_num_training_steps, build_callbacks


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg):
    cfg = setup(cfg)
    datamodule = LBDataModule(cfg)
    if False:
        dataset = datamodule._generate_dataset("val")
        print(len(dataset))
        dataloader = datamodule._generate_dataloader("val")
        print(len(dataloader))
        exit()
        for i in range(100):
            print(i)
            datamodule.train_dataloader()
        dataset = datamodule._generate_dataset("train")
        print(dataset[0])
        from tqdm.auto import tqdm
        dataloader = datamodule.train_dataloader()
        print(next(iter(dataloader)).keys())
        for batch in tqdm(dataloader):
            batch
        exit()
    elif False:
        model = LBModelModule(cfg)
        print(model)
        exit()
    num_train_data = len(datamodule._generate_dataset("train"))
    max_steps = get_num_training_steps(num_train_data, cfg)
    if "num_training_steps" in cfg.scheduler.params:
        cfg.scheduler.params.num_training_steps = max_steps
    if "T_max" in cfg.scheduler.params:
        cfg.scheduler.params.T_max = max_steps
    if "total_steps" in cfg.scheduler.params:
        cfg.scheduler.params.total_steps = max_steps
    modelmodule = LBModelModule(cfg)
    callbacks = build_callbacks(cfg)
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=WandbLogger(project="lb") if cfg.logger else None,
        **cfg.trainer,
    )
    trainer.fit(modelmodule, datamodule)


if __name__=="__main__":
    main()
