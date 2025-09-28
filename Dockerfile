from python:3.10.14-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*



COPY . /app/



COPY requirements.txt /app/


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 10000


CMD ["python", "app.py"]
