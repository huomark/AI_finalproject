notebook: main.py
	jupytext --to ipynb main.py

py: main.ipynb
	jupytext --to py:percent main.ipynb

flake: main.ipynb
	flake8 --show-source main.py

black: main.ipynb
	black --line-length 80 main.py