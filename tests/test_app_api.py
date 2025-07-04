import unittest
from unittest.mock import patch, mock_open
import app as web_app
from api import pdf_classifier_api as api_app

class TestWebApp(unittest.TestCase):
    def setUp(self):
        self.client = web_app.app.test_client()

    @patch('math_app.pdf_text_extractor.extract_text_from_pdf', return_value='test text')
    @patch('math_app.pdf_text_extractor.preprocess_text', return_value=['test', 'text'])
    @patch('sklearn.feature_extraction.text.CountVectorizer.transform', return_value=[[1, 2]])
    @patch('torch.tensor', return_value=None)
    @patch('math_app.text_classifier.SimpleNN.__call__', return_value=[[0.1, 0.9]])
    def test_upload_pdf(self, *mocks):
        data = {'pdf_file': (mock_open(read_data=b'data').return_value, 'test.pdf')}
        response = self.client.post('/', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Category', response.data)

class TestApiApp(unittest.TestCase):
    def setUp(self):
        self.client = api_app.app.test_client()

    @patch('math_app.pdf_text_extractor.extract_text_from_pdf', return_value='test text')
    @patch('math_app.pdf_text_extractor.preprocess_text', return_value=['test', 'text'])
    @patch('sklearn.feature_extraction.text.CountVectorizer.transform', return_value=[[1, 2]])
    @patch('torch.tensor', return_value=None)
    @patch('math_app.text_classifier.SimpleNN.__call__', return_value=[[0.1, 0.9]])
    def test_api_classify(self, *mocks):
        data = {'pdf_file': (mock_open(read_data=b'data').return_value, 'test.pdf')}
        response = self.client.post('/api/classify', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'category', response.data)

if __name__ == '__main__':
    unittest.main() 