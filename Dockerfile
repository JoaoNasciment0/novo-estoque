# Escolhe a imagem base com Python 3.11
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências (se existir)
COPY requirements.txt .

# Instala as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código-fonte da aplicação para o diretório de trabalho no container
COPY . .

# Expõe a porta que o Flask vai usar
EXPOSE 5000

# Define o comando para rodar a aplicação Flask
CMD ["python", "app.py"]
