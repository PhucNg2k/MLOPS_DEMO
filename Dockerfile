FROM apache/airflow:2.7.1-python3.10

# Switch to root to install system packages
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
# Switch back to airflow user for pip installations
USER airflow
RUN pip install --no-cache-dir --user -r /requirements.txt
RUN pip install awscli