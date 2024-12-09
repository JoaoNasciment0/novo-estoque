# Usar uma imagem base do Python 3.10
FROM python:3.10-slim

# Criar um diretório /app dentro do container
RUN mkdir /novo-estoque

# Definir /app como o diretório de trabalho dentro do container
WORKDIR /novo-estoque

# Copiar os arquivos da aplicação para o diretório /app no container
COPY . /novo-estoque

# Instalar as dependências do sistema
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

# Clonar o repositório YOLOv5 e instalar as dependências dele
RUN git clone https://github.com/ultralytics/yolov5.git /train/yolov5
WORKDIR /train/yolov5
RUN pip install --no-cache-dir -r requirements.txt

# Retornar ao diretório principal do projeto
WORKDIR /novo-estoque

# Garantir que o modelo treinado esteja no local correto
COPY IA-Treinada/best.pt IA-Treinada/best.pt

# Expor a porta que o Flask irá rodar
EXPOSE 5000

# Definir o comando para rodar o servidor Flask
CMD ["python", "main.py"]