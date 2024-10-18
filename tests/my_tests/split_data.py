import os
from equitrain.mace.tools.scripts_utils import get_dataset_from_xyz
from equitrain.mace.data.utils import random_train_valid_split

# Step 1: Define the dataset path
xyz_file = 'tests/data.xyz'

# Print the full path to verify access
full_path = os.path.abspath(xyz_file)
print(f"Trying to access file at: {full_path}")

# Step 2: Load the dataset (atomic energies dictionary and dataset)
atomic_energies_dict, dataset = get_dataset_from_xyz(
    xyz_file, valid_path=None, valid_fraction=0.2, config_type_weights=None
)

# Step 3: Split the dataset into training and validation using random_train_valid_split
valid_fraction = 0.2  # 20% of the data for validation
train_dataset, valid_dataset = random_train_valid_split(dataset, valid_fraction=valid_fraction, seed=42)

# Step 4: Save the split datasets into separate XYZ files
def save_xyz_data(output_file, dataset):
    with open(output_file, 'w') as f:
        for item in dataset:
            f.write(item)

# Define the paths for saving the split data
train_file = 'tests/my_tests/data/data_train.xyz'
valid_file = 'tests/my_tests/data/data_valid.xyz'

# Save the training and validation datasets
save_xyz_data(train_file, train_dataset)
save_xyz_data(valid_file, valid_dataset)

print(f"Training set saved to {train_file}")
print(f"Validation set saved to {valid_file}")