from pathlib import Path

import hydra
import numpy as np
import polars as pl

import lb.preprocessor


@hydra.main(config_path="conf", config_name="preprocess_data", version_base=None)
def main(cfg):
    # 既存ファイルのチェック
    basename = f"{cfg.preprocessor.prefix}_{cfg.stage}"
    if len(list(Path(cfg.output_dir).glob(f"{basename}_*.npy"))):
        if cfg.overwrite:
            print(f"overwrite: {basename}")
        else:
            print(f"already exists: {basename}")
            return
    # データ読み込み
    df = pl.read_parquet(
        Path(cfg.data_dir, f"processed_{cfg.stage}.parquet"),
        columns=cfg.use_cols,
    )
    df = df.with_columns(pl.int_range(len(df)).cast(pl.UInt32).alias("id"))
    if cfg.n_rows:
        df = df.sample(n=cfg.n_rows, seed=42)
    # 前処理の適用
    preprocessor = getattr(lb.preprocessor, cfg.preprocessor.name)(cfg)
    data_gen = preprocessor.apply(df)
    for i, data in enumerate(data_gen):
        np.save(Path(cfg.output_dir, f"{basename}_{i}.npy"), data)


if __name__=="__main__":
    main()
