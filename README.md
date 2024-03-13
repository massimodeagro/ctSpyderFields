# ctSpyderFields
Python package that project retinas through lenses as segmented from micro CT scans (Amira, Dragonfly) of spiders.
**[WARNING]**: This package is under developing. The code is unstable.

## How to install the package
**Step 0**: Download the package.
**ssh** (*suggested*):
```bash
git clone git@github.com:massimodeagro/ctSpyderFields.git
```
**https**:
```bash
git clone https://github.com/massimodeagro/ctSpyderFields.git
```

## Create your virtual environment:
**Step 1**: Create your virtual environment in your favorite path and activate it.
```bash
python -m venv ~/path/to/new/virtual/environment
source ~/path/to/new/virtual/environment/bin/activate
```
**Step 2**: Switch in the directory `ctSpyderFields`, then install the dependencies, using pip.
```bash
pip install -r requirements.txt
```
That's all folks! You can easily run the examples to test the package.
```bash
cd <ctSpyderFields_package_path>/ctSpyderFields/examples
python fullAnalysis.py
```