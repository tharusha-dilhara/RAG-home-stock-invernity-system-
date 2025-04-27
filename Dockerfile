FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy environment file first to leverage Docker cache
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "rag_project_env", "/bin/bash", "-c"]

# Copy application files
COPY . .

# Install additional Python dependencies using pip
RUN pip install pymongo==4.6.2 python-dateutil==2.8.2

# Ensure the environment is activated
ENV PATH /opt/conda/envs/rag_project_env/bin:$PATH

EXPOSE 7000

CMD ["conda", "run", "--no-capture-output", "-n", "rag_project_env", "python", "app.py"]
