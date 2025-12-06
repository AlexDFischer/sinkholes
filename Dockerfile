FROM condaforge/miniforge3:latest

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

WORKDIR /sinkholes
COPY . .

ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/bin/activate sinkholes && exec python3 ./sinkholes.py \"$@\"", "--"]
