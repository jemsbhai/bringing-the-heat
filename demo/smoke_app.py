"""Exercise the real localhost UI and prediction API, then close the server."""
import urllib.request
from gradio_client import Client
from app import app
from demo import RESULTS, write_json


if __name__ == "__main__":
    try:
        app.launch(server_name="127.0.0.1", server_port=7861, share=False,
                   prevent_thread_lock=True, quiet=True)
        status = urllib.request.urlopen("http://127.0.0.1:7861", timeout=30).status
        prediction = Client("http://127.0.0.1:7861", verbose=False).predict(
            "The Miami team won the championship after a dramatic final quarter.", api_name="/predict")
        assert status == 200
        assert prediction["label"] == "Sports", prediction
        write_json(RESULTS / "app_smoke.json", {"http_status": status, "prediction": prediction,
            "server": "127.0.0.1:7861", "public_share": False, "closed_after_test": True})
        print("Local HTTP page and prediction API passed; Sports returned.")
    finally:
        app.close()
