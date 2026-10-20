from denspart.cache import ComputeCache
from denspart.mbis import MBISProModel
from denspart.vh import optimize_reduce_pro_model


class Partitioning:
    def __init__(
        self, scheme, mol, moldens, grid, proatomdb, unique_atnums, pro_level, molecule_iodata=None
    ):
        self.scheme = scheme
        self.mol = mol
        self.moldens = moldens
        self.grid = grid
        self.proatomdb = proatomdb
        self.unique_atnums = unique_atnums
        self.pro_level = pro_level
        self.molecule_iodata = molecule_iodata
        self.pro_model = None
        self.localgrids = None

    def compute(self):
        scheme = self.scheme
        mol = self.mol
        moldens = self.moldens
        grid = self.grid
        # proatomdb = self.proatomdb

        # Partitioning scheme selection
        if scheme == "mbis":
            pro_model_init = MBISProModel.from_geometry(mol.atnums, mol.atcoords)
            pro_model, localgrids = optimize_reduce_pro_model(
                pro_model_init,
                grid,
                moldens,
                1e-8,
                1000,
                1e-20,
                ComputeCache(),
            )
        else:
            raise ValueError(f"Given scheme={scheme} not supported!")

        self.pro_model = pro_model
        self.localgrids = localgrids

        return pro_model, localgrids
