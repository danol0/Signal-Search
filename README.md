# Principles of Data Science Coursework, MT23

This repo contains the code and report that is submitted for the principles of data science coursework.

## Running the code
The code is written in Python. To install the required packages, run:
```bash
pip install -r requirements.txt
```
from the root directory of the repo. An environment.yml file is also included for conda users:
```bash
conda env create -f environment.yml
```
All scripts must be run from the root directory of the repo (do382) for the paths to work correctly.

The solutions to each question are in the `src` directory, and can be run individually, for example:
```bash
python src/solve_part_c.py
```
The code will run and save the plots to the report/figures directory.

### Parts (F) and (G)
The simulation studies take around 45 minutes to run, and will display progress bar. Data will save to the results directory once completed. Precomputed data is included in the results directory to allow plots to be generated without running the study - this can be achieved by setting the `run` parameter to `False` at the top of the `solve_part_f.py` and `solve_part_g.py` scripts.

## Report
The report is written in LaTeX and can be compiled by running:
```bash
pdflatex report.tex
```
A precompiled version of the report is included the report directory.