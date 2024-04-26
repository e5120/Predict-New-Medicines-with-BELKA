from pathlib import Path

import hydra
import numpy as np
import polars as pl

import lb.preprocessor


@hydra.main(config_path="conf", config_name="preprocess_data", version_base=None)
def main(cfg):
    filename = f"{cfg.preprocessor.prefix}_{cfg.stage}.npy"
    if Path(cfg.output_dir, filename).is_file():
        if cfg.overwrite:
            print(f"overwrite: {filename}")
        else:
            print(f"already exists: {filename}")
            return
    if not cfg.debug:
        cfg.n_rows = None
    df = pl.read_parquet(Path(cfg.data_dir, f"processed_{cfg.stage}.parquet"), n_rows=cfg.n_rows)
    preprocessor = getattr(lb.preprocessor, cfg.preprocessor.name)(cfg)
    data = preprocessor.apply(df)
    np.save(Path(cfg.output_dir, filename), data)


if __name__=="__main__":
    main()
