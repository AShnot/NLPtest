# PDF Text Classifier

**PDF Text Classifier** — это веб-приложение для автоматической классификации текстов из PDF-документов с помощью нейросети на PyTorch. Проект поддерживает запуск на GPU (CUDA) в Docker-контейнере, использует Flask для веб-интерфейса и Poetry для управления зависимостями.

---

## Возможности

- Загрузка PDF-файлов через веб-интерфейс.
- Извлечение текста из PDF.
- Классификация текста по 20 категориям (датасет 20 Newsgroups).
- Использование обученной модели и векторизатора.
- Поддержка работы на GPU (CUDA) через Docker.
- Совместимость с Poetry.
- **Веса модели и векторизатор хранятся в репозитории (папка `model/`).**

---

## Структура проекта

```
NLPtest/
├── app.py                  # Flask-приложение (веб-интерфейс)
├── math_app/
│   ├── text_classifier.py  # Модель, обучение, препроцессинг, инференс
│   ├── pdf_text_extractor.py # Извлечение и предобработка текста из PDF
│   └── ...                 # Прочие модули
├── config/
│   ├── settings.py         # Пути, настройки
│   └── logger.py           # Логирование
├── model/
│   ├── trained_model.pth   # Сохранённая PyTorch-модель 
│   └── vectorizer.pkl      # Сохранённый TfidfVectorizer 
├── classes/target_names.pkl # Список меток классов (будет в git)
├── templates/
│   └── index.html          # HTML-шаблон для Flask
├── pyproject.toml          # Poetry-зависимости
├── Dockerfile              # Docker-сборка с поддержкой CUDA
└── README.md               # Документация
```

---

## Быстрый старт

### 1. Клонируйте репозиторий

```bash
git clone <URL>
cd NLPtest
```

### 2. Обучите модель (локально или в контейнере)

```bash
poetry install
poetry run python math_app/text_classifier.py
```
- После обучения появятся файлы `model/trained_model.pth`, `model/vectorizer.pkl`, `classes/target_names.pkl`.
- **Или скачайте готовые веса из репозитория.**

### 3. Соберите Docker-образ

```bash
docker build -t nlpapp-gpu .
```

### 4. Запустите контейнер с поддержкой GPU

```bash
docker run --gpus all -p 5000:5000 nlpapp-gpu
```

### 5. Откройте веб-интерфейс

Перейдите в браузере по адресу: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Использование

1. Загрузите PDF-файл через форму.
2. Получите категорию текста (одна из 20 тем newsgroups).
3. Логи работы доступны в папке `logs/`.

---

## Переменные окружения

- `PORT` — порт для Flask/gunicorn (по умолчанию 5000).

---

## Обновление модели

1. Переобучите модель:
   ```bash
   poetry run python math_app/text_classifier.py
   ```
2. Перезапустите контейнер.

---

## Зависимости

- Python >=3.9,<3.12
- PyTorch (CUDA)
- Flask
- gunicorn (production-ready WSGI сервер)
- scikit-learn (совпадает с версией, на которой обучалась модель)
- nltk, PyPDF2, pydantic-settings, numpy

---

## Важно

- Для работы с GPU требуется установленный NVIDIA-драйвер и поддержка Docker GPU.
- Для корректной работы pickle-файлов версии scikit-learn должны совпадать между обучением и инференсом.
- В продакшене Flask запускается через gunicorn (см. Dockerfile).
- **Веса модели и векторизатор хранятся в репозитории для удобства развертывания.**
