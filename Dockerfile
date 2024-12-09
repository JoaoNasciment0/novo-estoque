# Usar uma imagem base do Python 3.10
FROM python:3.10-slim

# Criar um diretório de trabalho
WORKDIR /novo-estoque

# Copiar todos os arquivos da aplicação para o container
COPY . .

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalar as dependências do Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Instalar as dependências do YOLOv5
WORKDIR /novo-estoque/train/yolov5
RUN pip install --no-cache-dir -r requirements.txt

# Voltar para o diretório principal do projeto
WORKDIR /novo-estoque

# Expor a porta que o Flask irá rodar
EXPOSE 5000

CMD ["python", "main.py"]
