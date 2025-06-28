# 🌟 Production‑Ready AI/ML API

An end-to-end, production-ready API for deploying AI/ML models with full support for reproducibility, monitoring, testing, automation, and version control.

# Demo ▶️
You can find the demo here: [link](https://github.com/Akshay-66/production-ready-ai-api/blob/main/DEMO.md)

### Problem

* Data science models often fail to make the leap from research to production.
* Lack of reproducibility and poor testing/vetting processes lead to unreliable and unmaintainable deployments.
* Absence of monitoring and version control can cause performance degradation and rollback difficulties.
* Manual deployment processes introduce delays, errors, and inconsistencies.

---

### Solution

A robust API framework that wraps ML models into RESTful endpoints, with an integrated pipeline covering:

1. **Reproducibility**

   * Uses Docker, fixed seeds, and consistent data splits to ensure results can be reliably reproduced.

2. **Monitoring**

   * Live metrics (e.g. latency, throughput, error rates) exposed via Prometheus or similar tools.

3. **Testing**

   * Unit/integration tests for model logic and endpoint behavior.
   * Continuous validation using a holdout dataset for data drift detection.

4. **Automation**

   * Automated CI/CD pipelines triggered on every commit to build, test, and deploy.
   * Scheduled retraining + redeployment workflows for model updates.

5. **Version Control**

   * Git-managed model code, schema changes, and API definitions.
   * Tagged Docker images and deployment manifests ensure consistency across environments.

---

### Impact

* 💥 **Reliability**: Eliminates ad-hoc adoptions by ensuring every model meets strict testing and monitoring criteria.
* 🔁 **Faster Deployments**: With automated pipelines, new models go live in minutes instead of hours or days.
* 🔍 **Traceability**: Rollback is simple — we know exactly which model version is running and why.
* 📈 **Operational Visibility**: Real-time monitoring helps detect issues early and maintain performance SLAs.
* 🤝 **Collaborative Platform**: Data scientists, engineers, and ops teams can work from the same reproducible codebase.

---

### Architecture & Workflow

```text
[Data & Model Training] → [Dockerize + Test] → [CI/CD Pipeline] → [API Deployment]
       ↑                                                  ↓
      Model Registry ← Versioned Artifacts ← Monitoring & Alerts
```

---
### Tech Stack 
- **API Framework**: FastAPI
- **ML Framework**: scikit-learn, Pytorch
- **Containerization**: Docker
- **Testing**: pytest
- **Monitoring**: Prometheus
---

### Deployment 🚀

**Pre‑requisites**

* Docker & Docker Compose installed.
* Git, Python (with virtualenv), and Prometheus.
* CI/CD service (GitHub Actions).

**Steps**

1. **Clone**

   ```bash
   git clone https://github.com/Akshay-66/production-ready-ai-api.git
   cd production-ready-ai-api
   docker build -t ai-api:latest .
   ```
2.  **Build**
     ```bash
     docker build -t ai-api:latest
     ```
     or
    ```bash
    pip install -r requirements.txt
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
2. **Run tests**

   ```bash
   docker run ai-api:latest pytest
   ```
   or
    ```bash
   python -m pytest
   ```
   for Curl test of endpoint
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "I love this product!"}'
   ```

4. **Start locally**

   ```bash
   docker-compose up -d
   ```

   * Exposes API at `http://localhost:8000`
   * Monitoring metrics at `/metrics` (Prometheus-exporter)

5. **Configure CI/CD**

   * On every push:

     * Build Docker image, run tests.
     * Push image to registry (e.g. Docker Hub).
     * Update deployment (e.g. Docker Compose).
   * Use stages like **build → test → deploy → verify**.

6. **Model versioning**

   * Tag releases: `git tag v1.0.0`
   * Include model artifacts in image or load from S3/artifact store.
     
7. **Monitoring and alerts**

   * Scrape `/metrics` via Prometheus.

---

### Getting Started

1. Clone the repo.
2. Follow deployment steps above.
3. Test the API endpoints (e.g., `POST /predict`, `/metrics`).
4. Integrate CI/CD and monitoring using provided templates/configs.
### Note: Make sure you have `Git LFS` installed as it helps to push your large AI/ML Models.
### Git LFS - [https://git-lfs.com/]
---
