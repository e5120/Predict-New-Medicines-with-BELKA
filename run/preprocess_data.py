from pathlib import Path

import hydra
import numpy as np
import polars as pl

import lb.preprocessor
from lb.utils import PROTEIN_NAMES


@hydra.main(config_path="conf", config_name="preprocess_data", version_base=None)
def main(cfg):
    # 既存ファイルのチェック
    output_dir = Path(cfg.output_dir, cfg.preprocessor.prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = f"{cfg.preprocessor.prefix}_{cfg.stage}"
    exist_files = list(output_dir.glob(f"{basename}_*.npy"))
    if len(exist_files):
        if cfg.overwrite:
            print(f"overwrite: {basename}")
            for filename in exist_files:
                filename.unlink()
        else:
            print(f"already exists: {basename}")
            return
    # データ読み込み
    use_cols = cfg.preprocessor.use_cols
    if cfg.stage == "test":
        use_cols = list(filter(lambda x: x not in PROTEIN_NAMES, use_cols))
    df = pl.read_parquet(Path(cfg.data_dir, f"processed_{cfg.stage}.parquet"), columns=use_cols)
    if cfg.n_rows:
        df = df.sample(n=cfg.n_rows, seed=cfg.seed)
    df = df.select(pl.all().shuffle(seed=cfg.seed))
    # 前処理の適用
    preprocessor = getattr(lb.preprocessor, cfg.preprocessor.name)(cfg)
    data_gen = preprocessor.apply(df)
    for i, data in enumerate(data_gen):
        np.save(Path(output_dir, f"{basename}_{i}.npy"), data)


if __name__=="__main__":
    main()
