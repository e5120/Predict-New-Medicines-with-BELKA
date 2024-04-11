from pathlib import Path

import hydra
import polars as pl


SCHEMA = {
    "id": pl.Int32,
    "buildingblock1_smiles": pl.Utf8,
    "buildingblock2_smiles": pl.Utf8,
    "buildingblock3_smiles": pl.Utf8,
    "molecule_smiles": pl.Utf8,
    "protein_name": pl.Utf8,
    "binds": pl.Int32,
}


@hydra.main(config_path="conf", config_name="prepare_data", version_base=None)
def main(cfg):
    assert cfg.stage in ["train", "test"]
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.stage == "train":
        i = 0
        while True:
            df = pl.read_csv(Path(data_dir, "train.csv"), n_rows=cfg.n_rows, skip_rows=i*cfg.n_rows, schema=SCHEMA)
            df = df.with_columns(pl.col("binds").cast(pl.Boolean))
            start, end = df[0, "id"], df[-1, "id"]
            print(f"processed {start} ~ {end}")
            df.write_parquet(Path(output_dir, f"{cfg.prefix}train_{start}_{end}.parquet"))
            if len(df) != cfg.n_rows:
                break
            i += 1
    else:
        df = pl.read_parquet(Path(data_dir, "test.parquet"))
        df = df.with_columns(pl.col("id").cast(pl.Int32))
        df.write_parquet(Path(output_dir, f"{cfg.prefix}test.parquet"))


if __name__=="__main__":
    main()
