FROM continuumio/miniconda3

RUN mkdir app
WORKDIR /app

RUN mkdir report

COPY src src
COPY environment.yml .
COPY report/do382_PDS_report.tex report
COPY report/poweranalysis.bib report
COPY report/figures report/figures
COPY results results

RUN conda env update -f environment.yml --name base