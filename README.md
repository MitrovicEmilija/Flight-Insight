# ✈️ FlightInsight

> **Inteligentni sistem za napovedovanje letalskih zamud in analizo mnenj potnikov**

Projektna naloga pri predmetu **Inženirstvo inteligentnih sistemov (IIS)**.

---

## 📋 Opis projekta

FlightInsight je celovit ML sistem, ki implementira:

- 🛫 **Napoved zamude letov** — XGBoost regresor (lasten model), naučen na BTS Transtats podatkih (~7M letov iz 2024)
- 💬 **Sentiment analiza mnenj** — pretreniran HuggingFace RoBERTa transformer za klasifikacijo komentarjev potnikov
- 📊 **Analitika in vizualizacije** — interaktivni grafi v Plotly
- 🔧 **Admin nadzorna plošča** — monitoring, validacija (Great Expectations), drift (Evidently), MLflow tracking

Sistem zajema podatke iz **BTS Transtats** (uradni vir ZDA), jih validira, preprocesira in trenira model. Vse je avtomatizirano z **DVC pipeline-i** in **GitHub Actions CI/CD**. Aplikacija je pakirana v **Docker container**, ki je objavljen na **ghcr.io** (GitHub Container Registry).

### Tehnološki stack

| Kategorija | Tehnologije |
|------------|-------------|
| **Jezik + paketi** | Python 3.12, uv |
| **Podatki + verzioniranje** | BTS Transtats, DVC, DagsHub S3 |
| **Validacija + drift** | Great Expectations, Evidently AI |
| **ML modeli** | XGBoost, HuggingFace RoBERTa |
| **Eksperimenti** | MLflow (DagsHub backend) |
| **Uporabniški vmesnik** | Streamlit, Plotly |
| **CI/CD + Deployment** | GitHub Actions, Docker, ghcr.io |

---

## 🚀 Lokalni zagon

### Možnost 1: Docker (priporočeno — najhitreje)

Slika je predpripravljena z vsemi modeli in podatki. Potrebno je samo:

```bash
docker pull ghcr.io/mitrovicemilija/flight-insight:latest
docker run -p 8501:8501 ghcr.io/mitrovicemilija/flight-insight:latest
```

Aplikacija je dostopna na **http://localhost:8501**.

### Možnost 2: Docker Compose

```bash
git clone https://github.com/MitrovicEmilija/Flight-Insight.git
cd Flight-Insight
docker compose up -d
```

Ustavitev: `docker compose down`

### Možnost 3: Lokalni Python (za development)

**Predpogoji:**
- Python 3.12
- [uv](https://github.com/astral-sh/uv) paketni upravitelj
- Git

**Koraki:**

```bash
# 1. Clone repozitorij
git clone https://github.com/MitrovicEmilija/Flight-Insight.git
cd Flight-Insight

# 2. Namesti odvisnosti
uv sync

# 3. Pridobi modele in podatke iz DVC
uv run dvc pull

# 4. Zaženi Streamlit aplikacijo
uv run streamlit run frontend/app.py
```

Aplikacija je dostopna na **http://localhost:8501**.

---

## 🔗 Linki

| Vir | URL |
|-----|-----|
| **GitHub repozitorij** | https://github.com/MitrovicEmilija/Flight-Insight |
| **DagsHub repozitorij** (DVC + MLflow) | https://dagshub.com/MitrovicEmilija/Flight-Insight |
| **MLflow tracking UI** | https://dagshub.com/MitrovicEmilija/Flight-Insight.mlflow |
| **Docker slika** (ghcr.io) | https://github.com/MitrovicEmilija/Flight-Insight/pkgs/container/flight-insight |
| **ML Pipeline workflow** | https://github.com/MitrovicEmilija/Flight-Insight/actions/workflows/pipeline.yml |
| **Docker Build workflow** | https://github.com/MitrovicEmilija/Flight-Insight/actions/workflows/docker-build.yml |

---

## 📂 Struktura repozitorija

```
Flight-Insight/
├── .github/workflows/        # CI/CD pipeline-i (ML + Docker)
├── data/                     # Podatki (DVC tracked)
├── frontend/                 # Streamlit aplikacija
│   ├── app.py
│   ├── pages/                # 4 podstrani
│   └── utils/
├── gx/                       # Great Expectations konfiguracija
├── models/                   # Naučeni modeli (DVC)
├── reports/                  # HTML poročila (GX + Evidently)
├── scripts/                  # Pomožne skripte
├── src/
│   ├── data/                 # fetch, preprocess, drift detection
│   ├── model/                # XGBoost training
│   └── reviews/              # RoBERTa analyzer
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                  # DVC pipeline definicija
├── params.yaml               # Hiperparametri
├── pyproject.toml            # Python odvisnosti
└── README.md
```
