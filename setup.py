from setuptools import find_packages, setup

setup(
    name='equitrain',
    version='0.0.1',
    long_description='file: README.md',
    license='MIT',
    classifiers=[
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    include_package_data=True,
    packages=find_packages(include=['equitrain*']),
    install_requires=[
        'ase',
        'h5py',
        'numpy',
        'pymatgen',
        'torch',
        'torch_ema',
        'torch_geometric',
        'torch_scatter',
        'torch_cluster',
        'torchmetrics',
        'tqdm',
        'timm',
        'accelerate',
        'e3nn',
    ],
    python_requires='>=3.8',
    scripts=[
        'scripts/equitrain',
        'scripts/equitrain-preprocess',
    ],
)
