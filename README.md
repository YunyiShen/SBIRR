# Codebase for Multi-marginal Schrödinger Bridges with Iterative Reference Refinement
Code for 
Shen, Yunyi*, Renato Berlinghieri*, and Tamara Broderick. "Multi-marginal Schrödinger Bridges with Iterative Reference." arXiv preprint arXiv:2408.06277 (2024). *:equal contribution

## Structure
There are two main folders in this repo, `Notebook` and `package`. In the `package` we implmented our method as a python package `SBIRR` to be installed. After installation one can run experiments in `Notebook`

## Environment
To run our code we suggest the following procedure. 

1) Create a virtual environment with Python version Python 3.8.16, e.g., in conda `conda create -n "sbirr" python=3.8.16 ipython` and activate `conda activate sbirr` 
2) Install all dependencies needed from `requirements.txt`, by running `pip install -r requirements.txt`
3) Install our package SBIRR by running `pip install ./package` 

Then, to reproduce our experiment, run notebooks in `Notebooks`

## Experiments
Experiments using our method is written in noteboos in `Notebooks` folder, with name of each notebook being the experimental dataset. All cleaned data is in `Notebooks/data`
