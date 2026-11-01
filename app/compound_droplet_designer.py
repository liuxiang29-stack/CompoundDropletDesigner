"""Offline compound-droplet predictor and inverse-design application.

The executable starts a localhost-only HTTP server and opens the user's default
browser.  All model inference and optimization remain on the local computer.
No internet connection is used.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import webbrowser
import csv
import hashlib
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# Explicit imports help frozen builds reconstruct the serialized pipelines.
import sklearn.compose  # noqa: F401
import sklearn.ensemble  # noqa: F401
import sklearn.neural_network  # noqa: F401
import sklearn.pipeline  # noqa: F401
import sklearn.preprocessing  # noqa: F401
import model_runtime  # noqa: F401; required when loading the fitted offline pipelines
import conservation_corrected_runtime  # noqa: F401
import design_engine
from threadpoolctl import threadpool_limits


APP_NAME = "Compound Droplet Designer"
APP_VERSION = "1.5.2"
MODEL_VERSION = "ANN-soft-conservation-rechecked387-20261004"
HOST = "127.0.0.1"

CHANNEL_LENGTH = 5.0e-5
SIGMA_IM = 0.029
SIGMA_MO = 0.040
ETA_I = 0.109
ETA_O = 0.033
MODEL_CA_M_FACTOR = 1.0


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    # Research resources remain outside the public source checkout.
    private_root = os.environ.get("CDD_PRIVATE_RESOURCES")
    if private_root and relative.split("/", 1)[0] in {"data", "models"}:
        return Path(private_root).expanduser().resolve() / relative
    return base / relative


def writable_base() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


import tempfile
LOG_DIR = Path(tempfile.gettempdir()) / "CompoundDropletDesigner"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "compound_droplet_designer.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)


def features_from_raw(eta0: float, vi: float, vm: float, vo: float,
                      m: float, n_actual: float) -> np.ndarray:
    """Return displayed [Ca_i, Ca_o, Ca_m, De_m, n] in model column order."""
    return np.array([
        ETA_I * vi / SIGMA_IM,
        ETA_O * vo / SIGMA_MO,
        eta0 * vm / SIGMA_MO,
        m * vm / CHANNEL_LENGTH,
        n_actual,
    ], dtype=float)


def transformed_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float).copy()
    values[..., :4] = np.log10(np.maximum(values[..., :4], 1e-15))
    return values


@dataclass
class Candidate:
    vi: float
    vm: float
    vo: float
    ai: float
    am: float
    ai_error: float
    am_error: float
    combined_error: float
    support_score: float
    support_distance: float
    features: list[float]


class PredictorCore:
    def __init__(self) -> None:
        required_private = ["data/model_metadata.json", "data/stable_support.csv",
                            "models/soft_conservation_ann.joblib"]
        missing = [p for p in required_private if not resource_path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                "This source-only distribution does not include research datasets or fitted weights. "
                "Set CDD_PRIVATE_RESOURCES to an authorized private resource directory containing "
                "data/ and models/. Missing: " + ", ".join(missing))
        self.metadata = json.loads(resource_path("data/model_metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("model_version") != MODEL_VERSION:
            raise ValueError("Bundled model metadata does not match this application version.")
        if self.metadata.get("fitted_model_Ca_m_scale") != MODEL_CA_M_FACTOR:
            raise ValueError("The bundled Ca_m scaling does not match the trained models.")
        model_path = resource_path("models/soft_conservation_ann.joblib")
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != self.metadata["model_sha256"]:
            raise ValueError("The bundled model checksum does not match the approved model.")
        if self.metadata.get("ca_o_eta") != ETA_O:
            raise ValueError("The outer-phase viscosity does not match the bundled descriptor convention.")
        # The frozen networks were fitted in the historical Ca_o coordinate.
        # Correct its definition without refitting or changing physical-case predictions.
        self.fitted_ca_o_eta = float(self.metadata["fitted_model_ca_o_eta"])
        if not math.isfinite(self.fitted_ca_o_eta) or self.fitted_ca_o_eta <= 0:
            raise ValueError("Invalid fitted outer-capillary-number scale.")
        self.corrected_model = joblib.load(model_path)
        required = ["eta0", "vi", "vm", "vo", "m", "n_actual", "Ai", "Am"]
        with resource_path("data/stable_support.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or any(column not in rows[0] for column in required):
            raise ValueError("The bundled stable-support dataset is incomplete.")
        if len(rows) != self.metadata.get("support_cases"):
            raise ValueError("The bundled stable-support case count does not match the model metadata.")
        self.data = {
            column: np.array([float(row[column]) for row in rows], dtype=float)
            for column in required
        }

        self.stable_features = np.vstack([
            features_from_raw(eta0, vi, vm, vo, m, n_actual)
            for eta0, vi, vm, vo, m, n_actual in zip(
                self.data["eta0"], self.data["vi"], self.data["vm"],
                self.data["vo"], self.data["m"], self.data["n_actual"],
            )
        ])
        self.feature_min = self.stable_features.min(axis=0)
        self.feature_max = self.stable_features.max(axis=0)
        transformed = transformed_features(self.stable_features)
        q25, q75 = np.percentile(transformed, [25, 75], axis=0)
        scale = q75 - q25
        std = transformed.std(axis=0)
        self.feature_scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
        self.feature_center = np.median(transformed, axis=0)
        self.support_tree = cKDTree((transformed - self.feature_center) / self.feature_scale)

        self.raw_ranges = {
            c: [float(self.data[c].min()), float(self.data[c].max())]
            for c in ["eta0", "vi", "vm", "vo", "m", "n_actual"]
        }
        self.area_ranges = {
            "Ai": [float(self.data["Ai"].min() * 1e9), float(self.data["Ai"].max() * 1e9)],
            "Am": [float(self.data["Am"].min() * 1e9), float(self.data["Am"].max() * 1e9)],
        }
        # Use the 5th–95th percentile stable interior for inverse design.
        self.velocity_bounds = [
            tuple(map(float, np.quantile(self.data[c], [0.05, 0.95])))
            for c in ["vi", "vm", "vo"]
        ]
        self.search_lock = threading.Lock()
        self.nonstable_tree = None
        nonstable_path = resource_path('data/nonstable_features.csv')
        if nonstable_path.exists():
            with nonstable_path.open(encoding='utf-8-sig', newline='') as handle:
                records = list(csv.DictReader(handle))
            if records:
                a = np.array([[float(r[k]) for k in ('Ca_i','Ca_o','Ca_m','De_m','n')] for r in records])
                z = (transformed_features(a)-self.feature_center)/self.feature_scale
                self.nonstable_tree = cKDTree(z)
        self.nonstable_cases = self.nonstable_tree.n if self.nonstable_tree is not None else 0

    def nonstable_distance(self, features):
        x = np.atleast_2d(features)
        if self.nonstable_tree is None:
            return np.full(len(x), 1e6)
        z = (transformed_features(x)-self.feature_center)/self.feature_scale
        return np.asarray(self.nonstable_tree.query(z,k=1)[0])

    def predict_features(self, features: np.ndarray,
                         flow_ratio: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != 5 or not np.isfinite(x).all():
            raise ValueError("Five finite dimensionless inputs are required.")
        r = np.broadcast_to(np.asarray(flow_ratio, dtype=float), (len(x),))
        if not np.isfinite(r).all() or np.any(r <= 0):
            raise ValueError("A positive middle-to-inner inlet-flow ratio is required.")
        model_x = self.model_features(x)
        prediction = self.corrected_model.predict(np.column_stack((model_x, np.log(r))))
        return prediction[:, 0], prediction[:, 1]

    def model_features(self, features: np.ndarray) -> np.ndarray:
        """Map the corrected public descriptor convention to frozen fit coordinates."""
        x = np.asarray(features, dtype=float).copy()
        x[..., 1] *= self.fitted_ca_o_eta / ETA_O
        return x

    def support(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z = (transformed_features(x) - self.feature_center) / self.feature_scale
        distance, _ = self.support_tree.query(z, k=1)
        score = np.clip(100.0 * np.exp(-0.70 * distance), 0.0, 100.0)
        return np.asarray(distance), np.asarray(score)

    def assess_domain(self, features: np.ndarray) -> dict[str, Any]:
        f = np.asarray(features, dtype=float).ravel()
        outside = (f < self.feature_min) | (f > self.feature_max)
        distance, score = self.support(f)
        score_value = float(score[0])
        messages: list[str] = []
        labels = ["Ca_i", "Ca_o", "Ca_m", "De_m", "n"]
        for idx in np.flatnonzero(outside):
            messages.append(
                f"{labels[idx]} is outside the training range "
                f"[{self.feature_min[idx]:.4g}, {self.feature_max[idx]:.4g}]"
            )
        if outside.any() or score_value < 30:
            level = "red"
            headline = "Extrapolation: interpret this prediction with caution"
        elif score_value < 60:
            level = "yellow"
            headline = "Near a sparse or boundary region"
        else:
            level = "green"
            headline = "Within the main training-data region"
        if not messages:
            messages.append("All five inputs are within their observed univariate ranges.")
        return {
            "level": level,
            "headline": headline,
            "messages": messages,
            "support_score": score_value,
            "support_distance": float(distance[0]),
        }

    def forward(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode", "raw"))
        if mode == "raw":
            raw = {key: float(payload[key]) for key in ["eta0", "vi", "vm", "vo", "m", "n"]}
            self._validate_raw(raw)
            features = features_from_raw(raw["eta0"], raw["vi"], raw["vm"], raw["vo"], raw["m"], raw["n"])
            flow_ratio = 2.0 * raw["vm"] / raw["vi"]
        elif mode == "dimensionless":
            features = np.array([float(payload[key]) for key in ["cai", "cao", "cam", "dem", "n"]], dtype=float)
            if not np.isfinite(features).all() or np.any(features[:4] <= 0) or not 0 < features[4] < 1:
                raise ValueError("Ca_i, Ca_o, Ca_m, and De_m must be positive; n must be between 0 and 1.")
            raw = None
            flow_ratio = float(payload["flow_ratio"])
        else:
            raise ValueError("Unknown input mode.")

        ai, am = self.predict_features(features, flow_ratio)
        ai_value, am_value = float(ai[0]), float(am[0])
        vi_for_frequency = float(features[0]) * SIGMA_IM / ETA_I
        vm_for_frequency = flow_ratio * vi_for_frequency / 2
        tolerance = design_engine.finite(payload, 'consistency_tolerance_percent', 10, lo=.1, hi=100)
        fi, fm, frequency, gap, consistent = design_engine.frequency_metrics(
            ai_value, am_value, vi_for_frequency, vm_for_frequency, tolerance)
        return {
            "mode": mode,
            "raw": raw,
            "features": [float(v) for v in features],
            "flow_ratio": flow_ratio,
            "ai": ai_value,
            "am": am_value,
            "area_ratio": ai_value / am_value if am_value > 0 else None,
            "domain": self.assess_domain(features),
            "frequency": {
                "inner_hz": float(fi), "shell_hz": float(fm),
                "estimated_hz": float(frequency) if bool(consistent) else None,
                "consistency_percent": float(gap), "consistent": bool(consistent),
                "tolerance_percent": tolerance,
                "basis": "2D phase conservation; equal 50 um inlets; periodic single-core formation; negligible accumulation and bypass.",
                "status": "Estimated from predicted areas; not a direct detachment-time measurement."
            },
        }

    @staticmethod
    def _validate_raw(raw: dict[str, float]) -> None:
        if not all(math.isfinite(value) for value in raw.values()):
            raise ValueError("All inputs must be finite numbers.")
        if any(raw[key] <= 0 for key in ["eta0", "vi", "vm", "vo", "m"]):
            raise ValueError("Eta_0, all phase velocities, and m must be positive.")
        if not 0 < raw["n"] < 1:
            raise ValueError("n must be between 0 and 1.")

    def inverse(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.search_lock.acquire(blocking=False):
            raise ValueError("A design search is already running. Please wait for it to finish.")
        try:
            with threadpool_limits(limits=1):
                return design_engine.search(self, payload)
        finally:
            self.search_lock.release()

    @staticmethod
    def _features_for_velocities(velocities: np.ndarray, eta0: float, m: float, n: float) -> np.ndarray:
        v = np.asarray(velocities, dtype=float)
        return np.column_stack([
            ETA_I * v[:, 0] / SIGMA_IM,
            ETA_O * v[:, 2] / SIGMA_MO,
            eta0 * v[:, 1] / SIGMA_MO,
            m * v[:, 1] / CHANNEL_LENGTH,
            np.full(len(v), n),
        ])

    def model_info(self) -> dict[str, Any]:
        return {
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "model_version": MODEL_VERSION,
            "training_samples": int(self.metadata["training_cases"]),
            "support_samples": int(self.metadata["support_cases"]),
            "test_samples": int(self.metadata["test_cases"]),
            "correction_tau": float(self.metadata["projection_tau"]),
            "nonstable_observations": int(self.nonstable_cases),
            "velocity_bounds": {k:list(v) for k,v in zip(('vi','vm','vo'),self.velocity_bounds)},
            "design_objectives": ['area','max_frequency','target_frequency'],
            "frequency_units": 'Hz, based on 2D flux per unit depth and projected areas',
            "inputs": ["Ca_i", "Ca_o", "Ca_m", "De_m", "n"],
            "outputs": ["A_i", "A_m"],
            "raw_ranges": self.raw_ranges,
            "feature_ranges": {
                name: [float(lo), float(hi)]
                for name, lo, hi in zip(["Ca_i", "Ca_o", "Ca_m", "De_m", "n"], self.feature_min, self.feature_max)
            },
            "area_ranges": self.area_ranges,
            "validation": self.metadata["validation"],
            "validation_note": self.metadata["validation_note"],
            "constants": {
                "L": CHANNEL_LENGTH, "sigma_im": SIGMA_IM, "sigma_mo": SIGMA_MO,
                "eta_i": ETA_I, "eta_o": ETA_O,
            },
        }


class AppHandler(BaseHTTPRequestHandler):
    core: PredictorCore | None = None
    page: bytes = b""
    server_ref: ThreadingHTTPServer | None = None

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("HTTP " + fmt, *args)

    def _send_json(self, status: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(self.page)
        elif self.path == "/api/info":
            self._send_json(200, self.core.model_info())
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/forward":
                result = self.core.forward(payload)
                self._send_json(200, result)
            elif self.path == "/api/inverse":
                result = self.core.inverse(payload)
                self._send_json(200, result)
            elif self.path == "/api/shutdown":
                self._send_json(200, {"message": "Software closed."})
                threading.Thread(target=self.server_ref.shutdown, daemon=True).start()
            else:
                self.send_error(404)
        except Exception as exc:
            logging.exception("Request failed")
            self._send_json(400, {"error": str(exc)})


def show_fatal_error(message: str) -> None:
    logging.exception(message)
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)
    except Exception:
        pass


def main() -> None:
    try:
        threadpool_limits(limits=1)
        core = PredictorCore()
        page = resource_path("assets/index_v15.html").read_bytes()
        AppHandler.core = core
        AppHandler.page = page
        server = ThreadingHTTPServer((HOST, 0), AppHandler)
        AppHandler.server_ref = server
        port = server.server_address[1]
        url = f"http://{HOST}:{port}/"
        logging.info("Starting %s %s at %s", APP_NAME, APP_VERSION, url)
        if "--no-browser" not in sys.argv:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        server.serve_forever()
        server.server_close()
        logging.info("Application closed")
    except Exception as exc:
        show_fatal_error(f"Application failed to start: {exc}\n\nLog: {LOG_PATH}")
        raise


if __name__ == "__main__":
    main()
