# Usar uma imagem base do Python 3.10
FROM python:3.10-slim

# Criar um diretório /app dentro do container
RUN mkdir /app

# Definir /app como o diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos da aplicação para o diretório /app no container
COPY . /app

# Instalar as dependências do sistema (caso precise de bibliotecas nativas)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Instalar as dependências do Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expor a porta que o Flask irá rodar
EXPOSE 5000

# Definir o comando para rodar o servidor Flask
CMD ["python", "main.py"]
