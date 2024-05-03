import numpy as np
import polars as pl

from lb.preprocessor import BasePreprocessor


char2code = {'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10,
             '/': 11, 'c': 12, 'o': 13, '+': 14, 'I': 15, '5': 16, '(': 17, '2': 18, ')': 19,
             '9': 20, 'i': 21, '#': 22, '6': 23, '8': 24, '4': 25, '=': 26, '1': 27, 'O': 28, '[': 29, 'D': 30, 'B': 31, ']': 32, 'N': 33, '7': 34, 'n': 35, '-': 36}


def tokenize(smiles):
    output = [char2code[char] for char in smiles] + [0] * (142 - len(smiles))
    return output


class CharacterPreprocessor(BasePreprocessor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _apply(self, df):
        df = df.with_columns(
            pl.col("molecule_smiles").map_elements(tokenize).alias("input_ids"),
        )
        data = {}
        for i in range(len(df)):
            data[df[i, "id"]] = df[i, "input_ids"].to_numpy().astype(np.uint8)
        return data
