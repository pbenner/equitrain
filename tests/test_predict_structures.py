# %%

import ase.io

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from equitrain import get_args_parser_predict
from equitrain import get_model
from equitrain import predict_structures

# %%

def main():

    r = 4.5

    args = get_args_parser_predict().parse_args()

    args.load_checkpoint_model = f'result/best_val_epochs@1_e@255427.1345/pytorch_model.bin'

    model = get_model(5.0, args)

    atoms_list = ase.io.read('data.xyz', index=":")

    structures_list = [ AseAtomsAdaptor.get_structure(atom) for atom in atoms_list ]

    energy, force, stress = predict_structures(model, structures_list)

    print(energy)
    print()


# %%
if __name__ == "__main__":
    main()
