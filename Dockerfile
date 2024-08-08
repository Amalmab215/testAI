# Utiliser une image Python officielle comme base
FROM python:3.11.4

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt (si vous en avez un) et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application dans le conteneur
COPY . .

# Exposer le port sur lequel l'application sera accessible (modifiez-le si nécessaire)
EXPOSE 5000

# Spécifier la commande pour exécuter l'application
CMD ["python", "app.py"]
