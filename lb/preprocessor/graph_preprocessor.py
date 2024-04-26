import numpy as np
from rdkit import Chem

from lb.preprocessor import BasePreprocessor


class GraphPreprocessor(BasePreprocessor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.boolean = {True:1, False:0}
        self.hybridization = {'SP': 1, 'SP2': 2, 'SP3': 3, 'SP3D': 3.5}
        self.bondtype = {'SINGLE': 1, 'DOUBLE': 2, 'AROMATIC': 1.5, 'TRIPLE': 3}

    def _apply(self, df, data, start_idx):
        end_idx = start_idx + self.batch_size
        df = df.select(["id", "molecule_smiles"])
        for i in range(len(df[start_idx: end_idx])):
            smiles = df[i, "molecule_smiles"]
            mol = Chem.MolFromSmiles(smiles)
            node_attr = []
            for atom in mol.GetAtoms():
                node_attr.append(
                    [
                        atom.GetAtomicNum(),
                        atom.GetMass(),
                        self.hybridization[str(atom.GetHybridization())],
                        self.boolean[atom.IsInRing()],
                        self.boolean[atom.GetIsAromatic()],
                    ]
                )
            node_attr = np.array(node_attr)
            edges = []
            edge_attr = []
            for bond in mol.GetBonds():
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edge_attr.append(
                    [
                        self.bondtype[str(bond.GetBondType())],
                        self.boolean[bond.GetIsAromatic()],
                        self.boolean[bond.IsInRing()],
                    ]
                )
            edges = np.array(edges).T
            edge_attr = np.array(edge_attr)
            data[df[i, "id"]] = {
                "node_attr": node_attr,
                "edges": edges,
                "edge_attr": edge_attr,
            }
        return data
