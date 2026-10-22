# ctSpyderFields
Python package that project retinas through lenses as segmented from micro CT scans (Amira, Dragonfly) of spiders.
Requires Python 3.10, 3.11 or 3.12 (the package depends on numpy 1.x, which is not available for newer Python versions).


## How to install the package
**Step 0**: Make sure git is installed on your computer. On Ubuntu, you can do that with
```bash
sudo apt install git
```

### Option A: install with uv
[uv](https://docs.astral.sh/uv/) manages both Python and the package dependencies, and will automatically download a compatible Python version if needed.

**Step 1**: Install uv (see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for Windows and other options)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 2**: Create a virtual environment and install the `ctSpyderFields` package
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install git+https://github.com/massimodeagro/ctSpyderFields.git
```

Alternatively, if you are already working in a uv project, add the package to it with
```bash
uv add git+https://github.com/massimodeagro/ctSpyderFields.git
```

To reproduce the exact environment the package was tested with, clone the repository and sync from the lockfile (`uv.lock`)
```bash
git clone https://github.com/massimodeagro/ctSpyderFields.git
cd ctSpyderFields
uv sync
```

Since we suggest using the package interactively, you can start a Jupyter notebook in the environment with
```bash
uv run --with jupyter jupyter lab
```

### Option B: install with pip
**Step 1**: Create your virtual environment in your favorite path and activate it. Make sure to use Python 3.10, 3.11 or 3.12.
```bash
python -m venv ~/path/to/new/virtual/environment
source ~/path/to/new/virtual/environment/bin/activate
```

**Step 2**: Install the `ctSpyderFields` package using pip
```bash
pip install git+https://github.com/massimodeagro/ctSpyderFields.git
```

That's all folks! You can now run the examples to test the package. Sample data is available in the Data folder. Follow the guide at  [usage.md](https://github.com/massimodeagro/ctSpyderFields/blob/main/docs/usage.md).
To work on your own data follow the guide at  [data_preparation.md](https://github.com/massimodeagro/ctSpyderFields/blob/main/docs/data_preparation.md)

If you use this software, please reference it using the zenodo release: [![DOI](https://zenodo.org/badge/577233660.svg)](https://doi.org/10.5281/zenodo.18152995)


