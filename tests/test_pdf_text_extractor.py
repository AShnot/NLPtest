import unittest
from unittest.mock import patch, mock_open
from math_app import pdf_text_extractor

class TestPDFTextExtractor(unittest.TestCase):
    @patch('PyPDF2.PdfReader')
    def test_extract_text_from_pdf(self, mock_reader):
        # Мокаем поведение PyPDF2
        mock_page = type('MockPage', (), {'extract_text': lambda self: 'Hello PDF'})()
        mock_reader.return_value.pages = [mock_page, mock_page]
        with patch('builtins.open', mock_open(read_data=b'data')):
            text = pdf_text_extractor.extract_text_from_pdf('fake.pdf')
        self.assertEqual(text, 'Hello PDFHello PDF')

    def test_preprocess_text(self):
        text = 'Hello,   world! 123...'
        tokens = pdf_text_extractor.preprocess_text(text)
        self.assertEqual(tokens, ['Hello', 'world', '123'])

if __name__ == '__main__':
    unittest.main() 