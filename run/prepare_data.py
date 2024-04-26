from pathlib import Path
from multiprocessing import Pool

import hydra
import polars as pl
from tqdm.auto import tqdm
from rdkit import Chem


SCHEMA = {
    "id": pl.Int32,
    "buildingblock1_smiles": pl.Utf8,
    "buildingblock2_smiles": pl.Utf8,
    "buildingblock3_smiles": pl.Utf8,
    "molecule_smiles": pl.Utf8,
    "protein_name": pl.Utf8,
    "binds": pl.Int32,
}


def split_dataset(cfg):
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.stage == "train":
        i = 0
        while True:
            df = pl.read_csv(
                Path(data_dir, "train.csv"),
                n_rows=cfg.n_rows,
                skip_rows=i*cfg.n_rows,
                schema=SCHEMA,
            )
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
        df.write_parquet(Path(output_dir, f"{cfg.prefix}test_0_{len(df)}.parquet"))


def transform_dataset(df, cols=["BRD4", "HSA", "sEH"], is_test=False):
    if is_test:
        dataset_df = df.unique("molecule_smiles").drop(["id", "protein_name"])
    else:
        assert len(df) % 3 == 0
        dataset_df = None
        for i, protein_name in enumerate(cols):
            sub_df = df[i::3]
            sub_df = sub_df.rename({"binds": protein_name})
            if i == 0:
                dataset_df = sub_df.drop(["id", "protein_name"])
            else:
                dataset_df = pl.concat([dataset_df, sub_df[[protein_name]]], how="horizontal")
    dataset_df = dataset_df.rename(
        {
            "buildingblock1_smiles": "bb1_smiles",
            "buildingblock2_smiles": "bb2_smiles",
            "buildingblock3_smiles": "bb3_smiles",
        }
    )
    return dataset_df


def register_buildingblock(df, no, bb={}, is_test=False):
    smile_col = f"bb{no}_smiles"
    code_col = f"bb{no}_code"
    for smile, count in df[smile_col].value_counts().to_numpy():
        if smile not in bb:
            bb[smile] = {
                smile_col: smile,
                code_col: len(bb),
                "count": 0,
                "included_train": False,
                "included_test": False,
            }
        if is_test:
            bb[smile]["included_test"] = True
        else:
            bb[smile]["included_train"] = True
            bb[smile]["count"] += count
    return bb


def generate_bb_dataframe(bb):
    df = pl.from_dicts(list(bb.values()))
    df = df.with_columns(
        pl.col(pl.Int64).cast(pl.Int32),
    )
    return df


def transform_smiles(x):
    mol = Chem.MolFromSmiles(x)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return smiles


def generate_dataset(files, bb1={}, bb2={}, bb3={}, is_test=False):
    dfs = []
    for filename in tqdm(files):
        df = pl.read_parquet(filename)
        df = transform_dataset(df, is_test=is_test)
        with Pool() as p:
            non_isomeric_smiles = p.map(transform_smiles, df["molecule_smiles"].to_list())
            df = df.with_columns(
                pl.Series(non_isomeric_smiles).alias("non_isomeric_molecule_smiles"),
            )
        bb1 = register_buildingblock(df, 1, bb1, is_test=is_test)
        bb2 = register_buildingblock(df, 2, bb2, is_test=is_test)
        bb3 = register_buildingblock(df, 3, bb3, is_test=is_test)
        bb1_df = generate_bb_dataframe(bb1)[["bb1_smiles", "bb1_code"]]
        bb2_df = generate_bb_dataframe(bb2)[["bb2_smiles", "bb2_code"]]
        bb3_df = generate_bb_dataframe(bb3)[["bb3_smiles", "bb3_code"]]
        df = (
            df
            .join(bb1_df, on="bb1_smiles", how="inner")
            .join(bb2_df, on="bb2_smiles", how="inner")
            .join(bb3_df, on="bb3_smiles", how="inner")
            .drop(["bb1_smiles", "bb2_smiles", "bb3_smiles"])
        )
        dfs.append(df)
    df = pl.concat(dfs)
    df = df.with_columns(pl.int_range(len(df)).cast(pl.UInt32).alias("id"))
    return df, bb1, bb2, bb3


def aggregate_dataset(cfg):
    data_dir = Path(cfg.data_dir)
    trn_files = sorted(list(data_dir.glob(f"{cfg.prefix}train_*.parquet")))
    test_files = sorted(list(data_dir.glob(f"{cfg.prefix}test_*.parquet")))
    print(len(trn_files), len(test_files))
    trn_df, bb1, bb2, bb3 = generate_dataset(trn_files, is_test=False)
    test_df, bb1, bb2, bb3 = generate_dataset(test_files, bb1=bb1, bb2=bb2, bb3=bb3, is_test=True)
    trn_df.write_parquet(Path(cfg.output_dir, f"{cfg.prefix}train.parquet"))
    test_df.write_parquet(Path(cfg.output_dir, f"{cfg.prefix}test.parquet"))
    generate_bb_dataframe(bb1).write_parquet(Path(cfg.output_dir, f"{cfg.prefix}bb1.parquet"))
    generate_bb_dataframe(bb2).write_parquet(Path(cfg.output_dir, f"{cfg.prefix}bb2.parquet"))
    generate_bb_dataframe(bb3).write_parquet(Path(cfg.output_dir, f"{cfg.prefix}bb3.parquet"))


@hydra.main(config_path="conf", config_name="prepare_data", version_base=None)
def main(cfg):
    if cfg.phase == "split":
        assert cfg.stage in ["train", "test"]
        assert cfg.n_rows % 3 == 0
        split_dataset(cfg)
    elif cfg.phase == "aggregate":
        aggregate_dataset(cfg)
    else:
        raise NotImplementedError


if __name__=="__main__":
    main()
