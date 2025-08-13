import sys
from pathlib import Path
import requests
from datetime import datetime, timedelta

# Ensure project root is on sys.path for importing predict
sys.path.append(str(Path(__file__).resolve().parent.parent))
import predict


def _mock_requests_get():
    """Create a mock `requests.get` handling both Open-Meteo and SMHI calls."""

    def _get(url, params=None, timeout=10):
        if "open-meteo.com" in url and params and params.get("daily"):
            class MockResponse:
                status_code = 200

                def json(self):
                    tomorrow = datetime.now() + timedelta(days=1)
                    sunrise = tomorrow.replace(hour=6, minute=0, second=0, microsecond=0)
                    sunset = tomorrow.replace(hour=18, minute=0, second=0, microsecond=0)
                    return {
                        "daily": {
                            "sunrise": [sunrise.isoformat()],
                            "sunset": [sunset.isoformat()],
                        }
                    }

            return MockResponse()

        if "opendata-download-metfcst.smhi.se" in url:
            class MockResponse:
                status_code = 200

                def json(self):
                    forecast_time = (
                        datetime.now() + timedelta(days=1)
                    ).replace(hour=12, minute=0, second=0, microsecond=0).isoformat() + "Z"
                    return {
                        "timeSeries": [
                            {
                                "validTime": forecast_time,
                                "parameters": [
                                    {"name": "t", "values": [10]},
                                    {"name": "pis", "values": [0]},
                                    {"name": "pcat", "values": [0]},
                                    {"name": "pmean", "values": [0]},
                                    {"name": "tcc_mean", "values": [50]},
                                    {"name": "wsymb", "values": [1]},
                                ],
                            }
                        ]
                    }

            return MockResponse()

        raise AssertionError(f"Unexpected URL called: {url}")

    return _get


def test_get_smhi_forecast_returns_seven_values(monkeypatch):
    monkeypatch.setattr(predict, "get_radiation_forecast", lambda lat, lon: 1.0)
    monkeypatch.setattr(predict.requests, "get", _mock_requests_get())
    result = predict.get_smhi_forecast()
    assert isinstance(result, list)
    assert len(result) == 7


def test_get_smhi_forecast_handles_request_errors(monkeypatch):
    monkeypatch.setattr(predict, "get_radiation_forecast", lambda lat, lon: 1.0)

    def _raise(_url, params=None, timeout=10):
        raise requests.exceptions.RequestException("network error")

    monkeypatch.setattr(predict.requests, "get", _raise)
    assert predict.get_smhi_forecast() is None

