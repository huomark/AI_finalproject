# AI Final Project: Predicting Algorithm Tags for Competitive Programming Problems

## Using the Makefile

Ensure to install the following dependencies before proceeding:
- jupytext: For Jupyter notebook text conversion
- flake8: For code linting
- black: For code formatting

### Commands

- `make notebook`: Converts `main.py` to a Jupyter notebook (`main.ipynb`) using jupytext
- `make py`: Converts `main.ipynb` back to a Python script (`main.py`) using jupytext
- `make flake`: Runs flake8 linter on `main.py` to check for code quality issues
- `make black`: Formats `main.py` using black with a line length of 80 characters