from flask import Flask, jsonify
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

app = Flask(__name__)

REQUESTS = Counter(
    "ecdm_dependency_requests_total",
    "Total requests received by the dependency service",
)


@app.get("/")
def index():
    return jsonify(
        service="ECDM dependency service",
        status="running",
    )


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/value")
def value():
    REQUESTS.inc()

    return jsonify(
        status="ok",
        value=42,
    )


@app.get("/metrics")
def metrics():
    return (
        generate_latest(),
        200,
        {"Content-Type": CONTENT_TYPE_LATEST},
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8001,
        threaded=True,
    )
