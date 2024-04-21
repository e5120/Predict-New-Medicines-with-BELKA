from pathlib import Path

import hydra
import torch
import polars as pl
import lightning as L

from lb import LBDataModule, LBModelModule


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg):
    cfg.stage = "test"
    protein_names = ["BRD4", "HSA", "sEH"]
    datamodule = LBDataModule(cfg)
    test_dataloader = datamodule.test_dataloader()
    test_df = test_dataloader.dataset.df.select(["molecule_smiles"])
    trainer = L.Trainer(**cfg.trainer)
    if cfg.dir.name == "kaggle":
        model_paths = Path(cfg.dir.model_dir, f"hms-{cfg.exp_name}").glob("*.ckpt")
        output_dir = Path("/kaggle/working")
    else:
        model_paths = Path(cfg.dir.model_dir, cfg.exp_name, "single").glob("*.ckpt")
        output_dir = Path(cfg.dir.model_dir, cfg.exp_name, "single")
    for model_path in model_paths:
        print(model_path)
        modelmodule = LBModelModule.load_from_checkpoint(
            checkpoint_path=model_path,
            cfg=cfg,
        )
        predictions = trainer.predict(modelmodule, test_dataloader)
        predictions = torch.cat(predictions).numpy()
        pred_dfs = []
        for i, protein_name in enumerate(protein_names):
            pred_dfs.append(
                test_df.with_columns(
                    pl.lit(protein_name).alias("protein_name"),
                    pl.lit(predictions[:, i]).alias("binds"),
                )
            )
        pred_df = pl.concat(pred_dfs)
        submit_df = (
            pl.read_parquet(Path(cfg.data_dir, "test.parquet"), columns=["id", "molecule_smiles", "protein_name"])
            .join(pred_df, on=["molecule_smiles", "protein_name"], how="left")
            .select(["id", "binds"])
            .sort("id")
        )
        submit_df.write_csv(Path(output_dir, f"submission_{model_path.stem}.csv"))


if __name__=="__main__":
    main()
