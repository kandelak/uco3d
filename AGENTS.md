# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `uco3d/` (dataset loaders, frame builders, rotations, and utility helpers under `dataset_utils/` and `data_utils.py`).
- Dataset download logic is in `dataset_download/` with scripts and modality/category metadata (`category_to_archives.json`).
- Tests use `unittest` in `tests/`; heavier visualizations (Gaussian splats, point clouds) live in `test_gaussians_pcl.py`.
- Examples for API usage are in `examples/`. Media assets (`uco3d_logo.png`, `uco3d_grid.gif`) stay at the repo root.

## Setup, Build, and Development Commands
- Create a local venv (Python 3.10 recommended): `make install` (installs uv and project deps into `.venv`).
- Editable install for development: `pip install -e .` (run from the repo root).
- Set dataset root before running loaders: `export UCO3D_DATASET_ROOT=<DESTINATION_FOLDER>`.
- Download full dataset: `python dataset_download/download_dataset.py --download_folder <DESTINATION_FOLDER> --checksum_check`.
- Download via Hugging Face mirror: add `--use_huggingface`; download only specific modalities: `--download_modalities "rgb_videos,point_clouds"`.
- Small preview subset (~9.6 GB): `python dataset_download/download_dataset.py --download_small_subset --download_folder <SMALL_DEST>`.

## Testing Guidelines
- Run all tests: `cd tests && python run.py` (discovers `test_*.py` via `unittest`).
- To target a single test, edit the `if False` block in `tests/run.py` to the desired test path and set it to `True`.
- Visual tests (Gaussian splats/point clouds) require dataset samples and optional `gsplat`; skip them if assets are unavailable.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indent, snake_case for functions/variables, PascalCase for classes.
- Favor clear docstrings and type hints for public APIs (`UCO3DDataset`, `UCO3DFrameDataBuilder`).
- Keep module-level utilities small and focused; prefer adding helpers to existing utility files (`data_utils.py`, `dataset_utils/`) instead of scattering new ones.
- Avoid committing generated data or large artifacts; point to paths or environment variables instead.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative summaries (e.g., “add rot_utils from vggt” style seen in history); keep body focused on intent and impact.
- Before PRs: ensure tests pass, note any dataset prerequisites, and include reproduction commands.
- PR description should state scope, linked issues (if any), and whether behavior is user-facing (e.g., new download flags or dataset fields).
- Add screenshots/log snippets only when they clarify behavior; do not attach large binaries.***
