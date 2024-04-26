from pathlib import Path

import hydra
import numpy as np
import polars as pl

import lb.preprocessor


@hydra.main(config_path="conf", config_name="preprocess_data", version_base=None)
def main(cfg):
    # 既存ファイルのチェック
    filename = f"{cfg.preprocessor.prefix}_{cfg.stage}.npy"
    output_path = Path(cfg.output_dir, filename)
    if output_path.is_file():
        if cfg.overwrite:
            print(f"overwrite: {filename}")
        else:
            print(f"already exists: {filename}")
            return
    # データ読み込み
    df = pl.read_parquet(
        Path(cfg.data_dir, f"processed_{cfg.stage}.parquet"),
        columns=["non_isomeric_molecule_smiles"],
    )
    df = df.with_columns(pl.int_range(len(df)).cast(pl.UInt32).alias("id"))
    if cfg.n_rows:
        df = df.sample(n=cfg.n_rows, seed=42)
    # 前処理の適用
    preprocessor = getattr(lb.preprocessor, cfg.preprocessor.name)(cfg)
    data = preprocessor.apply(df)
    # 処理済みデータの保存
    np.save(output_path, data)


if __name__=="__main__":
    main()
