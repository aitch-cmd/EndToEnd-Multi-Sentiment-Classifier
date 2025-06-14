import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(text="I love this!"))

        self.assertEqual(response.status_code, 200)

        expected_labels = [b'sadness', b'joy', b'love', b'anger', b'fear']

        self.assertTrue(
            any(label in response.data for label in expected_labels),
            "Response should contain one of the emotion labels: sadness, joy, love, anger, or fear"
        )

if __name__ == '__main__':
    unittest.main()