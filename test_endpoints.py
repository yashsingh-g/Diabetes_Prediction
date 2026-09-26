"""
test_endpoints.py — Integration tests for the GlucoSense AI Flask application.

Run from the project root with the project's venv:
    python test_endpoints.py
    
Or explicitly:
    <project_root>/venv/Scripts/python.exe test_endpoints.py
"""
import unittest
import sys
import os

# Ensure the project root is on the path so `import app` resolves to
# <project_root>/app.py regardless of where this file is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Change working directory to project root so model.pkl is found on open()
os.chdir(PROJECT_ROOT)

from app import app  # noqa: E402  (import after sys.path manipulation is intentional)


class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    # ------------------------------------------------------------------
    # Home page
    # ------------------------------------------------------------------
    def test_home_page_get(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn('GlucoSense', content)
        self.assertIn('Patient Physiological Assessment', content)
        self.assertIn('name="glucose"', content)
        self.assertIn('name="bmi"', content)
        self.assertIn('name="pregnancies"', content)
        print('[PASS] Home page GET test passed.')

    # ------------------------------------------------------------------
    # GET /predict → no_value (redirect to error state)
    # ------------------------------------------------------------------
    def test_predict_get_renders_no_value(self):
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn('Biomarker Data Incomplete', content)
        print('[PASS] GET /predict renders no_value state.')

    # ------------------------------------------------------------------
    # POST /predict — low-risk profile (prediction == 0)
    # ------------------------------------------------------------------
    def test_predict_post_low_risk(self):
        response = self.client.post('/predict', data={
            'pregnancies': '1',
            'glucose': '85',
            'bloodpressure': '66',
            'skinthickness': '29',
            'insulin': '26',
            'bmi': '22.5',
            'dpf': '0.24',
            'age': '25',
        })
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn('Low Risk', content)
        self.assertNotIn('Biomarker Data Incomplete', content)
        print('[PASS] POST /predict low-risk profile => prediction=0 rendered.')

    # ------------------------------------------------------------------
    # POST /predict — high-risk profile (prediction == 1)
    # ------------------------------------------------------------------
    def test_predict_post_high_risk(self):
        response = self.client.post('/predict', data={
            'pregnancies': '5',
            'glucose': '168',
            'bloodpressure': '84',
            'skinthickness': '36',
            'insulin': '180',
            'bmi': '35.8',
            'dpf': '0.65',
            'age': '52',
        })
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn('Elevated Risk', content)
        self.assertNotIn('Biomarker Data Incomplete', content)
        print('[PASS] POST /predict high-risk profile => prediction=1 rendered.')

    # ------------------------------------------------------------------
    # POST /predict — missing fields → graceful no_value error page
    # ------------------------------------------------------------------
    def test_predict_post_missing_fields(self):
        response = self.client.post('/predict', data={'glucose': '100'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Biomarker Data Incomplete', response.data)
        print('[PASS] POST /predict missing fields => no_value error page.')

    # ------------------------------------------------------------------
    # POST /predict — non-numeric input → graceful no_value error page
    # ------------------------------------------------------------------
    def test_predict_post_invalid_input(self):
        response = self.client.post('/predict', data={
            'pregnancies': 'not-a-number',
            'glucose': '168',
            'bloodpressure': '84',
            'skinthickness': '36',
            'insulin': '180',
            'bmi': '35.8',
            'dpf': '0.65',
            'age': '52',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Biomarker Data Incomplete', response.data)
        print('[PASS] POST /predict invalid (non-numeric) input => no_value error page.')

    # ------------------------------------------------------------------
    # POST /predict — empty form → graceful no_value error page
    # ------------------------------------------------------------------
    def test_predict_post_empty_form(self):
        response = self.client.post('/predict', data={})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Biomarker Data Incomplete', response.data)
        print('[PASS] POST /predict empty form => no_value error page.')

    # ------------------------------------------------------------------
    # POST /predict — boundary values (zeros are technically valid inputs
    # for some fields in the Pima dataset, Flask must not 500)
    # ------------------------------------------------------------------
    def test_predict_post_boundary_zeros(self):
        response = self.client.post('/predict', data={
            'pregnancies': '0',
            'glucose': '40',  # min valid glucose
            'bloodpressure': '30',  # min valid BP
            'skinthickness': '0',
            'insulin': '0',
            'bmi': '10.0',  # min valid BMI
            'dpf': '0.05',
            'age': '1',
        })
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        # Should render a result (not an error)
        self.assertNotIn('Biomarker Data Incomplete', content)
        print('[PASS] POST /predict boundary/zero values => result rendered (no 500).')


if __name__ == '__main__':
    print('=' * 60)
    print('GlucoSense AI — Flask Endpoint Test Suite')
    print('=' * 60)
    unittest.main(verbosity=2)
