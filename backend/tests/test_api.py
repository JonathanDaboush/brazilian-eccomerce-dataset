import unittest

from fastapi.testclient import TestClient

from main import app


class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        response = self.client.post("/api/replay/reset")
        self.assertEqual(response.status_code, 200)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("event_bank", payload)
        self.assertTrue(payload["source_data_present"])

    def test_replay_populates_dashboard(self):
        replay = self.client.post("/api/replay", json={"start_offset": 0, "limit": 25, "pace_ms": 0})
        self.assertEqual(replay.status_code, 200)
        dashboard = self.client.get("/api/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        payload = dashboard.json()
        self.assertGreater(payload["kpis"]["orders"], 0)
        self.assertGreaterEqual(len(payload["activity"]), 1)

    def test_models_endpoint(self):
        response = self.client.get("/api/ml/models")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(len(payload["models"]), 1)


if __name__ == "__main__":
    unittest.main()
