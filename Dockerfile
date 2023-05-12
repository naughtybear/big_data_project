FROM apache/airflow:2.6.0
COPY requirements.txt /requirements.txt
RUN pip3 install --user --upgrade pip
RUN pip3 install --no-cache-dir --user -r /requirements.txt
