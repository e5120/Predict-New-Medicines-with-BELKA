from pathlib import Path
from multiprocessing import Pool

import hydra
import polars as pl
from tqdm.auto import tqdm
from rdkit import Chem

from lb.utils import PROTEIN_NAMES


SCHEMA = {
    "id": pl.Int32,
    "buildingblock1_smiles": pl.Utf8,
    "buildingblock2_smiles": pl.Utf8,
    "buildingblock3_smiles": pl.Utf8,
    "molecule_smiles": pl.Utf8,
    "protein_name": pl.Utf8,
    "binds": pl.Int32,
}


def smiles2code(df ,s2c):
    for i in range(1, 4):
        bb_name = f"bb{i}_smiles"
        uniq_smiles = df[bb_name].unique().to_list()
        for smiles in uniq_smiles:
            if smiles not in s2c:
                s2c[smiles] = len(s2c)
        df = (
            df
            .with_columns(pl.col(bb_name).replace(s2c).alias(f"bb{i}_code").cast(pl.Int16))
            .drop(bb_name)
        )
    return df


def transform_dataset(df, cols=PROTEIN_NAMES, is_test=False):
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


def split_dataset(cfg):
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    s2c = {}
    # test data
    print("processed test data")
    test_df = pl.read_parquet(Path(data_dir, "test.parquet"))
    test_df = transform_dataset(test_df, is_test=True)
    test_df = smiles2code(test_df, s2c)
    test_df.write_parquet(Path(output_dir, f"{cfg.prefix}test_0_{len(test_df)}.parquet"))
    # train data
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
        df = transform_dataset(df)
        df = smiles2code(df, s2c)
        df.write_parquet(Path(output_dir, f"{cfg.prefix}train_{start}_{end}.parquet"))
        if len(df) != cfg.n_rows // 3:
            break
        i += 1
    # bb smiles to code
    bb_df = pl.DataFrame({"bb_smiles": s2c.keys(), "bb_code": s2c.values()})
    bb_df = bb_df.with_columns(pl.col("bb_code").cast(pl.Int16))
    bb_df.write_parquet(Path(output_dir, f"{cfg.prefix}bb.parquet"))


def transform_smiles(x):
    mol = Chem.MolFromSmiles(x)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return smiles


def load_dataset(files):
    dfs = []
    for filename in tqdm(files):
        df = pl.read_parquet(filename)
        with Pool() as p:
            non_isomeric_smiles = p.map(transform_smiles, df["molecule_smiles"].to_list())
            df = df.with_columns(
                pl.Series(non_isomeric_smiles).alias("non_isomeric_molecule_smiles"),
            )
        dfs.append(df)
    df = pl.concat(dfs)
    df = df.with_columns(pl.int_range(len(df)).cast(pl.UInt32).alias("id"))
    return df


def aggregate_dataset(cfg):
    data_dir = Path(cfg.data_dir)
    trn_files = sorted(list(data_dir.glob(f"{cfg.prefix}train_*.parquet")))
    test_files = sorted(list(data_dir.glob(f"{cfg.prefix}test_*.parquet")))
    trn_df = load_dataset(trn_files)
    test_df = load_dataset(test_files)
    trn_df.write_parquet(Path(cfg.output_dir, f"{cfg.prefix}train.parquet"))
    test_df.write_parquet(Path(cfg.output_dir, f"{cfg.prefix}test.parquet"))


def cross_validation(cfg):
    # ToDo: foldごとに正解ラベルの分布が違いすぎるのが問題ないか
    df = pl.read_parquet(Path(cfg.data_dir, f"{cfg.prefix}train.parquet"))
    bb_count = []
    for i in range(1, 4):
        bb_count.append(
            df[f"bb{i}_code"].value_counts().rename({f"bb{i}_code": "bb_code"})
        )
    bb_count_df = (
        pl.concat(bb_count)
        .group_by("bb_code")
        .agg(pl.col("count").sum())
        .with_columns((pl.col("count").rank(method="ordinal") % cfg.n_folds).alias("fold"))
    )
    bb_count_df.write_parquet(Path(cfg.data_dir, f"{cfg.prefix}fold_info.parquet"))
    for fold in range(cfg.n_folds):
        trn_df = (
            df
            .filter(~pl.col("bb1_code").is_in(bb_count_df.filter(pl.col("fold") == fold)["bb_code"]))
            .filter(~pl.col("bb2_code").is_in(bb_count_df.filter(pl.col("fold") == fold)["bb_code"]))
            .filter(~pl.col("bb3_code").is_in(bb_count_df.filter(pl.col("fold") == fold)["bb_code"]))
        )
        non_share_val_df = (
            df
            .filter(pl.col("bb1_code").is_in(bb_count_df.filter(pl.col("fold") == fold)["bb_code"]))
            .filter(pl.col("bb2_code").is_in(bb_count_df.filter(pl.col("fold") == fold)["bb_code"]))
            .filter(pl.col("bb3_code").is_in(bb_count_df.filter(pl.col("fold") == fold)["bb_code"]))
        )
        share_val_df = trn_df.sample(n=int(0.725*len(non_share_val_df)), seed=cfg.seed)
        val_df = pl.concat(
            [
                share_val_df.with_columns(pl.lit(False).alias("non-share")),
                non_share_val_df.with_columns(pl.lit(True).alias("non-share")),
            ],
        )
        print(f"[fold {fold}]")
        print(f"# of non-share data: {len(non_share_val_df)}")
        print(non_share_val_df[PROTEIN_NAMES].sum())
        print(f"# of share data: {len(share_val_df)}")
        print(share_val_df[PROTEIN_NAMES].sum())
        val_df.write_parquet(Path(cfg.data_dir, f"{cfg.prefix}val_fold{fold}.parquet"))


@hydra.main(config_path="conf", config_name="prepare_data", version_base=None)
def main(cfg):
    if cfg.output_dir is None:
        cfg.output_dir = cfg.data_dir
    if cfg.phase == "split":
        assert cfg.n_rows % 3 == 0
        split_dataset(cfg)
    elif cfg.phase == "aggregate":
        aggregate_dataset(cfg)
    elif cfg.phase == "cross_validation":
        cross_validation(cfg)
    else:
        raise NotImplementedError


if __name__=="__main__":
    main()
