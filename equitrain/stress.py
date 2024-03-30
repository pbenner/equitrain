import torch

from typing import List, Optional, Tuple

def compute_stress_virials(
    energy: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:

    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]

    virials = torch.autograd.grad(
        outputs      = [energy],       # [n_graphs, ]
        inputs       = [displacement], # [n_nodes, 3]
        grad_outputs = grad_outputs,
        retain_graph = training,       # Make sure the graph is not destroyed during training
        create_graph = training,       # Create graph for second derivative
        allow_unused = True,
    )
    stress = torch.zeros_like(displacement)

    cell = cell.view(-1, 3, 3)
    volume = torch.einsum(
        "zi,zi->z",
        cell[:, 0, :],
        torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
    ).unsqueeze(-1)
    stress = virials / volume.view(-1, 1, 1)

    return stress


def get_displacement(
    positions: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:

    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype  = positions.dtype,
        device = positions.device,
    )
    displacement.requires_grad_(True)

    return displacement

def compute_stress(data, energy, training=False) -> torch.Tensor:

    displacement = get_displacement(
        positions=data["positions"],
        num_graphs=num_graphs,
    )

    stress = compute_stress_virials(
        energy=energy,
        displacement=displacement,
        cell=data["cell"],
        training=training,
    )

    return stress
