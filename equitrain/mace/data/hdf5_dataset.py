import h5py

from ase import Atoms
from torch.utils.data import Dataset, IterableDataset, ChainDataset

from equitrain.ocpmodels.preprocessing import AtomsToGraphs

class CachedCalc:

    def __init__(self, energy, forces, stress):
        self.energy = energy
        self.forces = forces
        self.stress = stress

    def get_potential_energy(self, apply_constraint=False):
        return self.energy

    def get_forces(self, apply_constraint=False):
        return self.forces

    def get_stress(self, apply_constraint=False):
        return self.stress

class HDF5ChainDataset(ChainDataset):
    def __init__(self, file_path, r_max, z_table, **kwargs):
        super(HDF5ChainDataset, self).__init__()
        self.file_path = file_path
        self._file = None

        self.length = len(self.file.keys())
        self.r_max = r_max
        self.z_table = z_table

    @property
    def file(self):
        if self._file is None:
            # If a file has not already been opened, open one here
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self):
        _d = dict(self.__dict__)
        
        # An opened h5py.File cannot be pickled, so we must exclude it from the state
        _d["_file"] = None
        return _d

    def __call__(self):
        datasets = []
        for i in range(self.length):
            grp = self.file["config_" + str(i)]
            datasets.append(
                HDF5IterDataset(
                    iter_group=grp,
                    r_max=self.r_max,
                    z_table=self.z_table,
                )
            )
        return ChainDataset(datasets)


class HDF5IterDataset(IterableDataset):
    def __init__(self, iter_group, r_max, z_table, **kwargs):
        super(HDF5IterDataset, self).__init__()
        # it might be dangerous to open the file here
        # move opening of file to __getitem__?
        self.iter_group = iter_group
        self.length = len(self.iter_group.keys())
        self.converter = AtomsToGraphs(r_energy=True, r_forces=True, r_stress=True, radius=r_max)
        self.r_max = r_max
        self.z_table = z_table

    def __len__(self):
        return self.length

    def __iter__(self):
        grp = self.iter_group
        len_subgrp = len(grp.keys())
        grp_list = []
        for i in range(len_subgrp):
            subgrp = grp["config_" + str(i)]

            atoms = Atoms(
                numbers   = subgrp["atomic_numbers"][()],
                positions = subgrp["positions"][()],
                cell      = subgrp["cell"][()],
                pbc       = subgrp["pbc"][()],
            )
            atoms.calc = CachedCalc(
                subgrp["energy"][()],
                subgrp["forces"][()],
                subgrp["stress"][()],
            )
            graphs = self.converter.convert(atoms)

            grp_list.append(graphs)

        return iter(grp_list)

class HDF5Dataset(Dataset):
    def __init__(self, file_path, r_max, z_table, **kwargs):
        super(HDF5Dataset, self).__init__()
        self.file_path = file_path
        self._file = None
        batch_key = list(self.file.keys())[0]
        self.batch_size = len(self.file[batch_key].keys())
        self.length = len(self.file.keys()) * self.batch_size
        self.converter = AtomsToGraphs(r_energy=True, r_forces=True, r_stress=True, radius=r_max)
        self.r_max = r_max
        self.z_table = z_table
        try:
            self.drop_last = bool(self.file.attrs["drop_last"])
        except KeyError:
            self.drop_last = False

    @property
    def file(self):
        if self._file is None:
            # If a file has not already been opened, open one here
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self):
        _d = dict(self.__dict__)

        # An opened h5py.File cannot be pickled, so we must exclude it from the state
        _d["_file"] = None
        return _d

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # compute the index of the batch
        batch_index = index // self.batch_size
        config_index = index % self.batch_size
        grp = self.file["config_batch_" + str(batch_index)]
        subgrp = grp["config_" + str(config_index)]

        atoms = Atoms(
            numbers   = subgrp["atomic_numbers"][()],
            positions = subgrp["positions"][()],
            cell      = unpack_value(subgrp["cell"][()]),
            pbc       = unpack_value(subgrp["pbc"][()]),
        )
        atoms.calc = CachedCalc(
            unpack_value(subgrp["energy"][()]),
            unpack_value(subgrp["forces"][()]),
            unpack_value(subgrp["stress"][()]),
        )
        graphs = self.converter.convert(atoms)

        return graphs


def unpack_value(value):
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value
