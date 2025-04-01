FROM ubuntu:latest
RUN apt-get update -y && apt-get install -y git wget
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && chmod +x Miniforge3-Linux-x86_64.sh && ./Miniforge3-Linux-x86_64.sh -b
RUN . /root/miniforge3/bin/activate \
	&& conda create -n sinkholes python=3.11 \
	&& conda activate sinkholes \
	&& conda install -c conda-forge richdem
COPY . ./sinkholes
WORKDIR sinkholes
RUN . /root/miniforge3/bin/activate \
	&& conda activate sinkholes \
	&& pip install -r requirements.txt
ENTRYPOINT ["/bin/bash", "-c", "source /root/miniforge3/bin/activate sinkholes && exec python3 ./sinkholes.py \"$@\"", "--"]
