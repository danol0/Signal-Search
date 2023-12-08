# Principles of Data Science Coursework, MT23

This repo contains the code and report that is submitted for the principles of data science coursework.

## Running the code
The code is written in Python 3.11.5. To install the required packages, run:
```bash
pip install -r requirements.txt
```
from the root directory of the repo. I have also included an environment.yml file for conda users:
```bash
conda env create -f environment.yml
```

The solutions to each question are in the `src` directory, and can be run individually, for example:
```bash
python src/solve_part_c.py
```
The code will run and save the plots to the report/figures directory.
If a simulation is required, a tqdm progress bar will show and the data will save to the results directory once completed.

## Report
The report is written in LaTeX and can be compiled by running:
```bash
pdflatex report.tex
```
from the root directory of the repo. The report will be saved as `report.pdf` in the root directory.
I have also included a precompiled version of the report in the report directory.