# 🎬 Movie Recommendation System

A full end-to-end machine learning project that builds a **personalized movie recommendation system** using a two-stage **Retrieval + Ranking** architecture, trained on the MovieLens 1M dataset, and deployed with **FastAPI** + **Streamlit**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)

---

## Overview

This project implements a production-style recommendation system inspired by how large platforms like YouTube and Netflix serve personalized content. The system works in two stages:

1. **Retrieval** — quickly finds ~200 candidate movies for a user using a Two-Tower neural network and cosine similarity
2. **Ranking** — re-ranks the candidates using a deep neural network that predicts the exact rating a user would give each movie

---

## Dataset

**MovieLens 1M** — a classic benchmark dataset for recommendation systems.

| File | Description | Size |
|------|-------------|------|
| `ratings.dat` | user_id · movie_id · rating (1–5) · timestamp | 1,000,209 ratings |
| `movies.dat` | movie_id · title · genres | 3,952 movies |
| `users.dat` | user_id · gender · age · occupation · zip | 6,040 users |

**Key stats:**
- Rating range: 1 – 5 stars
- 56.8% of ratings are 4 or 5 stars (positive skew)
- 18 unique genres (pipe-separated per movie)
- Data split: **80% train / 20% test** (chronological — sorted by timestamp)

---

## Project Structure

```
recommendationSysetms/
├── main.py                        ← FastAPI backend
├── streamlit_app.py               ← Streamlit frontend
├── recommendation_system.ipynb    ← Training notebook
├── requirements.txt               ← Python dependencies
├── models/                        ← Saved artifacts
│   ├── user_tower.keras
│   ├── movie_tower.keras
│   ├── ranking_model.keras
│   ├── retrieval_model.keras
│   ├── all_movie_vectors.npy
│   ├── encoders.pkl
│   ├── year_scaler.pkl
│   ├── genre_cols.pkl
│   ├── movie_full.parquet
│   └── users.parquet
└── ml-1m/                         ← Raw dataset
    ├── ratings.dat
    ├── movies.dat
    └── users.dat
```

---

## Pipeline

```
Raw Data
   │
   ▼
Exploratory Data Analysis (EDA)
   │  • Rating distribution
   │  • Top rated / most rated movies
   │  • User demographics
   ▼
Preprocessing & Feature Engineering
   │  • Sequential ID encoding (users & movies)
   │  • Genre one-hot encoding (18 genres)
   │  • Release year extraction + MinMax scaling
   │  • User features: gender, age group, occupation
   │  • Chronological train/test split (80/20)
   ▼
Stage 1 — Retrieval Model (Two-Tower)
   │  • User Tower: embeddings → Dense(64) → L2 normalize
   │  • Movie Tower: embeddings + genre + year → Dense(64) → L2 normalize
   │  • Dot product → Sigmoid → Binary cross-entropy loss
   │  • Output: Top-200 candidate movies per user
   ▼
Stage 2 — Ranking Model
   │  • Input: [user_vector(64) ‖ movie_vector(64)] = 128 features
   │  • Dense(256) → BN → Dropout → Dense(128) → BN → Dropout → Dense(64) → Linear
   │  • MSE loss → predicts exact rating (1–5)
   ▼
Saved Artifacts
   │  • Models (.keras), encoders (.pkl), vectors (.npy), dataframes (.parquet)
   ▼
Deployment
   • FastAPI  → REST API (port 8000)
   • Streamlit → Web UI  (port 8501)
```

---

## Model Architecture

### Retrieval Model — Two-Tower Network

```
User Tower                          Movie Tower
──────────────────────────          ──────────────────────────
Embedding(user_id,   32)            Embedding(movie_id, 32)
Embedding(gender,     8)            Genre OHE (18 features)
Embedding(age,        8)            Year scaled (1 feature)
Embedding(occupation, 8)                    │
        │                                   │
   Concatenate                         Concatenate
        │                                   │
   Dense(64, ReLU)                    Dense(64, ReLU)
   Dropout(0.2)                       Dropout(0.2)
   Dense(64, ReLU)                    Dense(64, ReLU)
        │                                   │
  UnitNormalization               UnitNormalization
        │                                   │
        └──────── Dot Product ──────────────┘
                       │
                  Dense(1, Sigmoid)
                       │
               Binary Cross-Entropy
```

### Ranking Model — Deep Neural Network

```
Input: [user_vector(64) ‖ movie_vector(64)] → 128 features
         │
   Dense(256, ReLU)
   BatchNormalization
   Dropout(0.3)
         │
   Dense(128, ReLU)
   BatchNormalization
   Dropout(0.2)
         │
   Dense(64, ReLU)
   Dropout(0.1)
         │
   Dense(1, Linear)   ← predicted rating
         │
       MSE Loss
```

---

## Results

| Model | Metric | Value |
|-------|--------|-------|
| Retrieval | AUC | > 0.75 |
| Ranking | RMSE | ~0.85 |
| Ranking | MAE | ~0.67 |

**Training config:**
- Optimizer: Adam (lr = 1e-3)
- Batch size: 2,048
- Early stopping: patience = 3 (monitoring val metric)
- ReduceLROnPlateau: factor = 0.5, patience = 2 (ranking only)

---

## Installation

```bash
# 1. Clone or download the project
cd recommendationSysetms

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

You need **two terminals** open simultaneously.

### Terminal 1 — FastAPI Backend
```bash
uvicorn main:app --reload --port 8000
```
- API base: http://localhost:8000
- Interactive docs: http://localhost:8000/docs

### Terminal 2 — Streamlit Frontend
```bash
streamlit run streamlit_app.py
```
- Web UI: http://localhost:8501

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/stats` | System statistics |
| `GET` | `/user/{user_id}` | User profile |
| `GET` | `/movie/{movie_id}` | Movie details |
| `POST` | `/recommend` | Get recommendations |

### Example — POST `/recommend`

**Request:**
```json
{
  "user_id": 1,
  "top_k_retrieve": 200,
  "top_n_final": 10
}
```

**Response:**
```json
{
  "user_id": 1,
  "gender": "F",
  "age": 18,
  "occupation": 10,
  "recommendations": [
    {
      "rank": 1,
      "movie_id": 1265,
      "title": "Groundhog Day (1993)",
      "genres": "Comedy|Romance",
      "year": 1993,
      "ranking_score": 4.52,
      "retrieval_score": 0.9823
    }
  ]
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Deep Learning | TensorFlow 2.20 / Keras 3.13 |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Serialization | Pickle, Joblib, Parquet |
| Environment | Virtual Environment (venv) |

---

## Notes

- User IDs range from **1 to 6,040**
- First API startup takes ~30 seconds (models loading into memory)
- Movie vectors are **pre-computed** at startup for fast O(N) inference
- The `Lambda` layer in original models was replaced with `UnitNormalization` for Keras 3 serialization compatibility
