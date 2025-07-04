from config.logger import get_logger
import nltk
nltk.download('stopwords')
from config.settings import settings
logger = get_logger(__name__)
from flask import Flask, request, render_template
import os
from math_app.pdf_text_extractor import extract_text_from_pdf
import torch
from math_app.text_classifier import ImprovedNN, advanced_preprocess
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Загружаем метки классов
with open(settings.CLASSES, 'rb') as f:
    target_names = pickle.load(f)

# Загружаем обученный vectorizer
if settings.VECTOR_PATH.exists():
    with open(settings.VECTOR_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    logger.info(f"Загружен обученный vectorizer из {settings.VECTOR_PATH}")
else:
    logger.warning(f"Файл vectorizer {settings.VECTOR_PATH} не найден. Используется новый vectorizer.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=20000)

# Загружаем обученную модель
# Выбор устройства
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedNN(input_dim=20000, num_classes=len(target_names)).to(DEVICE)
if settings.MODEL_PATH.exists():
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=DEVICE))
    logger.info(f"Загружена предобученная модель из {settings.MODEL_PATH}")
else:
    logger.warning(f"Файл модели {settings.MODEL_PATH} не найден. Используется не обученная модель.")
model.eval()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    if request.method == 'POST':
        file = request.files['pdf_file']
        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            logger.info(f"Загружен файл: {filepath}")
            try:
                text = extract_text_from_pdf(filepath)
                processed_text = advanced_preprocess(text)
                X_vec = vectorizer.transform([processed_text]).toarray()
                X_tensor = torch.tensor(X_vec, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    outputs = model(X_tensor)
                    _, predicted = torch.max(outputs, 1)
                    category = target_names[predicted.item()]
                prediction = category
                logger.info(f"Файл {filepath} классифицирован как: {category}")
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {filepath}: {e}")
                prediction = f"Ошибка: {e}"
        else:
            logger.warning("Загружен не-PDF файл или файл не выбран")
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)