from config.logger import get_logger
logger = get_logger(__name__)
import PyPDF2
import re

def extract_text_from_pdf(pdf_path):
    """
    Извлекает текст из PDF-файла по указанному пути.
    :param pdf_path: str, путь к PDF-файлу
    :return: str, извлечённый текст
    """
    logger.info(f"Извлечение текста из PDF: {pdf_path}")
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        logger.info(f"Текст успешно извлечён из {pdf_path}")
    except Exception as e:
        logger.error(f"Ошибка при извлечении текста из {pdf_path}: {e}")
        raise
    return text