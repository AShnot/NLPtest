# ============================
# Импорты и инициализация
# ============================
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from config.logger import get_logger
from config.settings import settings
logger = get_logger(__name__)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import pickle
from math_app.pdf_text_extractor import extract_text_from_pdf

# ============================
# Препроцессинг текста
# ============================
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def advanced_preprocess(text):
    """
    Выполняет базовый препроцессинг текста:
    - Приводит к нижнему регистру
    - Удаляет все символы, кроме латинских букв и пробелов
    - Токенизирует
    - Лемматизирует
    - Удаляет стоп-слова и короткие слова (<=2 символа)
    :param text: str, исходный текст
    :return: str, обработанный текст
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # только буквы
    text = re.sub(r'\s+', ' ', text)
    tokens = text.strip().split(' ')
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token and token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

# ============================
# Датасет для PyTorch
# ============================
class TextDataset(Dataset):
    """
    Кастомный датасет для текстовых данных.
    Принимает векторизованные тексты и метки классов.
    """
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ============================
# Архитектура нейросети
# ============================
class ImprovedNN(nn.Module):
    """
    Улучшенная нейросеть для классификации текстов.
    Содержит 4 полносвязных слоя, BatchNorm и Dropout для регуляризации.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# ============================
# Обучение модели
# ============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_text_classifier():
    """
    Обучает нейросетевую модель на датасете 20 Newsgroups.
    Сохраняет лучшую модель и векторизатор. Early stopping при отсутствии улучшения.
    """
    try:
        # ====== Загрузка и препроцессинг данных ======
        logger.info("Загрузка датасета 20 Newsgroups")
        data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        logger.info("Улучшенная предобработка текста")
        X = [advanced_preprocess(text) for text in data.data]
        X_train, X_test, y_train, y_test = train_test_split(X, data.target, test_size=0.2, random_state=42)

        # ====== Векторизация текста ======
        logger.info("TF-IDF векторизация текста (20000 признаков)")
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        X_train_vec = vectorizer.fit_transform(X_train).toarray()
        X_test_vec = vectorizer.transform(X_test).toarray()

        # ====== Подготовка датасетов и загрузчиков ======
        train_dataset = TextDataset(X_train_vec, y_train)
        test_dataset = TextDataset(X_test_vec, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # ====== Инициализация модели, функции потерь, оптимизатора и scheduler ======
        model = ImprovedNN(input_dim=20000, num_classes=len(data.target_names)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        logger.info("Начало обучения модели (50 эпох)")
        best_acc = 0
        epochs_no_improve = 0
        early_stop_patience = 7  # Количество эпох без улучшения для early stopping
        max_epochs = 50

        # ====== Основной цикл обучения ======
        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} complete. Средний loss: {avg_loss:.4f}")

            # ====== Валидация на тестовой выборке ======
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            accuracy = 100 * correct / total
            logger.info(f"Validation accuracy: {accuracy:.2f}%")
            scheduler.step(accuracy)

            # ====== Сохранение лучшей модели ======
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), settings.MODEL_PATH)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # ====== Early stopping ======
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping: нет улучшения {early_stop_patience} эпох подряд. Останавливаем обучение.")
                break

        # ====== Финальное сохранение векторизатора и меток классов ======
        with open(settings.VECTOR_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(settings.CLASSES, 'wb') as f:
            pickle.dump(data.target_names, f)

        # ====== Финальный лог и вывод ======
        logger.info(f"Test accuracy: {best_acc:.2f}% (best)")
        print(f"Test accuracy: {best_acc:.2f}% (best)")
        logger.info(f"Модель сохранена в {settings.MODEL_PATH}")
        logger.info(f"Векторизатор сохранён в {settings.VECTOR_PATH}")
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        raise

# ============================
# Предсказание класса
# ============================
def predict_class(processed_text, vectorizer, model):
    """
    Предсказывает класс для уже обработанного текста (строка).
    :param processed_text: str, препроцессированный текст
    :param vectorizer: обученный TfidfVectorizer
    :param model: обученная модель
    :return: int, номер класса
    """
    X_vec = vectorizer.transform([processed_text]).toarray()
    X_tensor = torch.tensor(X_vec, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# ============================
# Классификация PDF-документов
# ============================
def classify_pdf(pdf_path):
    """
    Классифицирует PDF-документ по его содержимому.
    :param pdf_path: str, путь к PDF-файлу
    :return: int, номер предсказанного класса
    """
    logger.info(f"Классификация PDF: {pdf_path}")
    # Загрузка модели и векторизатора
    model = ImprovedNN(input_dim=20000, num_classes=20).to(DEVICE)  # 20 классов для 20newsgroups
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=DEVICE))
    model.eval()
    with open(settings.VECTOR_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    # Извлечение текста
    text = extract_text_from_pdf(pdf_path)
    # Препроцессинг и токенизация
    processed_text = advanced_preprocess(text)
    # Предсказание
    predicted = predict_class(processed_text, vectorizer, model)
    logger.info(f"PDF классифицирован как класс: {predicted}")
    return predicted

# ============================
# Точка входа
# ============================
if __name__ == "__main__":
    train_text_classifier() 