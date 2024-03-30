from setuptools import setup

setup(name='equitrain',
      version='0.0.1',
      long_description='file: README.md',
      license='MIT',
      classifiers=[
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
      ],
      include_package_data=True,
      packages=['equitrain'],
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
        ],
      python_requires='>=3.8',
      scripts=[
          'scripts/equitrain',
          'scripts/equitrain-preprocess',
        ]
      )
