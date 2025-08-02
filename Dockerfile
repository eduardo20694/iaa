FROM python:3.10-slim

WORKDIR /app

# Atualiza e instala build-essential só se necessário (pode evitar se não compilar libs)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Atualiza pip e instala dependências
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY . .

# Usa 1 worker e timeout maior para economizar memória e evitar timeout (ajuste para seu ambiente)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "120", "app:app"]
