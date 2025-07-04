# Используем официальный PyTorch-образ с поддержкой CUDA
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Системные зависимости
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Poetry
ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Копируем pyproject.toml
WORKDIR /app
COPY pyproject.toml ./

# Устанавливаем зависимости через poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Копируем остальной проект
COPY . .

# Скачиваем NLTK ресурсы
RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Открываем порт для Flask
EXPOSE 5000

# Запуск приложения через gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 