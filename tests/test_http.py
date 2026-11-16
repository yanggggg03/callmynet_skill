import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from server import Handler
from core import ROOT


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.service.config_path = str(ROOT / "config/models.json")
        cls.thread = threading.Thread(target=cls.service.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.service.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.service.shutdown()
        cls.service.server_close()
        cls.thread.join()

    def test_health_and_example_are_accessible_without_weights(self):
        health = json.load(urllib.request.urlopen(self.url + "/api/health"))
        self.assertEqual(health["service"], "callmynet-skill")
        demo = json.load(urllib.request.urlopen(self.url + "/api/demo"))
        self.assertEqual(demo["mode"], "demo")

    def test_untrusted_origin_cannot_invoke_model(self):
        request = urllib.request.Request(self.url + "/api/predict", data=b"{}",
                                         headers={"Origin": "https://unrelated.example", "Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_templates_are_xlsx(self):
        response = urllib.request.urlopen(self.url + "/api/templates/waveforms.xlsx")
        self.assertEqual(response.read(2), b"PK")

    def test_private_config_is_not_served(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.url + "/config/models.json")
        self.assertEqual(error.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
