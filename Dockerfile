FROM python:3.12

# Dipendenze di sistema per compilazione ed esecuzione

RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Librerie Python per Quantum Machine Learning e Data Science

RUN pip install --no-cache-dir \
    ipython \
    numpy \
    scipy \
    matplotlib \
    qiskit \
    qiskit-aer \
    pennylane \
    jupyterlab \
    notebook \
    scikit-learn \
    pandas \
    pennylane-lightning \
    seaborn \
    tabulate \
    tqdm \
    openpyxl \
    joblib

# Variabili d'ambiente per il multithreading
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
