# mae-data-gen

Traditional numerical solvers for generating Monge-Ampère equation reference data.

## Structure

```
src/mae_data_gen/
├── solvers/          # Numerical solvers (FDM, FEM, etc.)
├── problems/         # Problem definitions (domain, f, g)
├── mesh.py           # Mesh generation utilities
└── io.py             # Data export (.npz / .hdf5)
scripts/
└── generate.py       # CLI entry point
```

## Usage

```bash
pip install -e .
python scripts/generate.py --problem unit_disk --solver fdm --resolution 128
```
