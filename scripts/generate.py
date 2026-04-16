"""Hydra CLI for generating Monge-Ampere equation training data."""

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from mae_data_gen.io import save_dataset
from mae_data_gen.mesh import grid_disk, sample_disk, sample_disk_boundary
from mae_data_gen.problems import create_problem


def _generate_split(
    problem,
    n_interior: int,
    n_boundary: int,
    rng: np.random.Generator,
    output_dir: Path,
    split_name: str,
    resolution: int | None = None,
) -> None:
    """Generate and save one data split (train/val/test).

    Args:
        problem: Problem instance with analytical solutions.
        n_interior: Number of random interior points (ignored when *resolution* is set).
        n_boundary: Number of boundary points.
        rng: Random generator for reproducibility.
        output_dir: Directory to write output files.
        split_name: Name prefix for the output files (e.g. ``"train"``).
        resolution: If set, use a grid with this many points per axis instead of random sampling.
    """
    if resolution is not None:
        x_int = grid_disk(resolution)
    else:
        x_int = sample_disk(n_interior, rng=rng)

    u_int = problem.exact_solution(x_int)
    grad_u_int = problem.exact_gradient(x_int)
    hess_u_int = problem.exact_hessian(x_int)
    det_hess_int = np.linalg.det(hess_u_int)
    f_int = problem.rhs(x_int)

    save_dataset(
        output_dir / f"{split_name}_interior",
        x_int,
        {
            "u": u_int,
            "grad_u": grad_u_int,
            "hessian_u": hess_u_int,
            "det_hessian": det_hess_int,
            "f": f_int,
        },
    )

    x_bnd = sample_disk_boundary(n_boundary)
    u_bnd = problem.boundary_value(x_bnd)
    save_dataset(
        output_dir / f"{split_name}_boundary",
        x_bnd,
        {"u_boundary": u_bnd},
    )

    n_pts = len(x_int)
    print(f"  [{split_name}] interior={n_pts}, boundary={n_boundary} -> {output_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for data generation."""
    problem = create_problem(cfg.problem.name)
    rng = np.random.default_rng(cfg.sampling.seed)
    output_dir = Path(cfg.output.dir)

    resolution = cfg.sampling.get("resolution", None)

    print(f"Generating {cfg.problem.name} data -> {output_dir}")
    for split in cfg.sampling.splits:
        _generate_split(
            problem,
            n_interior=cfg.sampling.n_interior,
            n_boundary=cfg.sampling.n_boundary,
            rng=rng,
            output_dir=output_dir,
            split_name=split,
            resolution=resolution,
        )
    print("Done.")


if __name__ == "__main__":
    main()
