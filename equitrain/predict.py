import numpy as np
import torch
import torch_geometric

from typing import List

from torch_geometric.data.collate import collate

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from equitrain.model       import get_model
from equitrain.dataloaders import get_dataloader
from equitrain.ocpmodels.preprocessing import AtomsToGraphs

# %%

def predict_structures(model: torch.nn.Module, structure_list: List[Structure], max_neighbors=200, r_max = None, num_workers = 1, pin_memory = False, batch_size = 12, device = None) -> List[torch.Tensor]:
    """Predict energy, forces, and stress of a structure"""

    if hasattr(model, 'cutoff'):
        r_max = model.cutoff
    if hasattr(model, 'max_radius'):
        r_max = model.max_radius

    if r_max is None:
        raise ValueError('Could not determine r_max value')

    atoms_to_graphs = AtomsToGraphs(
        max_neigh=max_neighbors,
        radius=r_max,
        r_energy=False,
        r_forces=False,
        r_stress=False,
        r_distances=True,
        r_edges=True,
        r_fixed=False,
        r_pbc=True,
    )

    data = [ atoms_to_graphs.convert(AseAtomsAdaptor.get_atoms(structure)) for structure in structure_list ]

    data_loader = torch_geometric.loader.DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    r_energy = torch.empty((0), device=device)
    r_force  = torch.empty((0, 3), device=device)
    r_stress = torch.empty((0, 3, 3), device=device)

    for datum in data_loader:

        datum = datum.to(device)

        energy, force, stress = model(datum)

        r_energy = torch.cat((r_energy, energy), dim=0)
        r_force  = torch.cat((r_force , force ), dim=0)
        r_stress = torch.cat((r_stress, stress), dim=0)

    return r_energy, r_force, r_stress


def predict_structure(model: torch.nn.Module, structure: Structure, max_neighbors=200, r_max = None, device = None) -> List[torch.Tensor]:
    """Predict energy, forces, and stress of a structure"""

    if hasattr(model, 'cutoff'):
        r_max = model.cutoff
    if hasattr(model, 'max_radius'):
        r_max = model.max_radius

    if r_max is None:
        raise ValueError('Could not determine r_max value')

    atoms_to_graphs = AtomsToGraphs(
        max_neigh=max_neighbors,
        radius=r_max,
        r_energy=False,
        r_forces=False,
        r_stress=False,
        r_distances=True,
        r_edges=True,
        r_fixed=False,
        r_pbc=True,
    )

    atoms = AseAtomsAdaptor.get_atoms(structure)
    data = atoms_to_graphs.convert(atoms)
    data = data.to(device)

    data, _, _ = collate(data.__class__, [data])

    return model(data)


def _predict(args):

    energy_pred = []
    forces_pred = [[], [], []]
    stress_pred = [[], [], [], [], [], [], [], [], []]

    data_loader, r_max = get_dataloader(args.predict_file, args)

    model = get_model(r_max, args)

    for step, data in enumerate(data_loader):

        e_pred, f_pred, s_pred = model(data)

        if e_pred is not None:
            energy_pred.extend([ e.item() for e in e_pred ])

        if f_pred is not None:

            for i in range(3):

                forces_pred[i].extend([ f[i].item() for f in f_pred ])

        if s_pred is not None:

            for i in range(3):

                for j in range(3):

                    k = i*3 + j

                    stress_pred[k].extend([ s[i][j].item() for s in s_pred ])

    return np.array(energy_pred), np.array(forces_pred), np.array(stress_pred)

def predict(args):

    if args.predict_file is None:
        raise ValueError("--predict-file is a required argument")
    if args.statistics_file is None:
        raise ValueError("--statistics-file is a required argument")
    if args.load_checkpoint_model is None:
        raise ValueError("--load-checkpoint-model is a required argument")

    return _predict(args)