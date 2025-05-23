#!/bin/bash

echo "🔧 Atualizando pacotes..."
sudo apt update -y && sudo apt upgrade -y

echo "📦 Instalando dependências do sistema..."
sudo apt install -y python3 python3-venv python3-pip nginx unzip

echo "📁 Criando diretório do projeto em /opt/fingerprint-app"
sudo mkdir -p /opt/fingerprint-app
sudo chown $USER:$USER /opt/fingerprint-app

echo "📂 Copiando arquivos do projeto..."
unzip BiomatchML-main.zip -d /opt/
cd /opt/fingerprint-app

echo "🐍 Criando ambiente virtual..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Instalando dependências Python..."
pip install --upgrade pip
pip install -r requirements.txt

echo "📁 Verificando estrutura da base de dados..."
mkdir -p database/imagens

echo "📥 Copie suas imagens .bmp para o diretório: /opt/fingerprint-app/database/imagens"
read -p "Pressione Enter para continuar após adicionar as imagens..."

echo "🧠 Gerando base vetorial de embeddings..."
python3 create_database.py

echo "🔐 Configurando serviço Gunicorn..."
sudo cp deploy/gunicorn.service /etc/systemd/system/gunicorn.service
sudo systemctl daemon-reexec
sudo systemctl start gunicorn
sudo systemctl enable gunicorn

echo "🌐 Configurando NGINX..."
sudo cp deploy/nginx.conf /etc/nginx/sites-available/fingerprint
sudo ln -sf /etc/nginx/sites-available/fingerprint /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

echo "✅ Deploy finalizado com sucesso. Acesse via IP ou domínio configurado."
