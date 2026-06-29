FROM python:3.12-slim

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
    openpyxl 
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]