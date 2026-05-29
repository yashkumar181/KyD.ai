from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import os
import uuid
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from models import Base, Dataset

# Load environment variables
load_dotenv()

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing.")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Robust connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       
    pool_recycle=300,         
    pool_size=5,              
    max_overflow=10,          
    connect_args={
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# FastAPI Setup
app = FastAPI(title="KyD.ai API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

def calculate_health_score(df: pd.DataFrame) -> int:
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    duplicate_pct = df.duplicated().sum() / df.shape[0]
    score = 100 - (missing_pct * 40) - (duplicate_pct * 20)
    return max(0, int(score))

# --- ENDPOINT 1: THE UPLOAD ROUTE ---
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, f"{file_id}_{file.filename}")

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

    rows, cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    health_score = calculate_health_score(df)
    
    missing_summary = [
        {"column": col, "missing_pct": round((df[col].isnull().sum() / rows) * 100, 2), "dtype": str(df[col].dtype)}
        for col in df.columns if df[col].isnull().any()
    ]

    eda_data = {
        "missing_summary": missing_summary,
        "duplicate_count": int(df.duplicated().sum())
    }

    db = SessionLocal()
    try:
        new_dataset = Dataset(
            id=file_id,
            filename=file.filename,
            file_path=file_path,
            rows=rows,
            columns=cols,
            memory_mb=round(memory_mb, 2),
            health_score=health_score,
            eda_json=eda_data
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

    return {
        "dataset_id": str(new_dataset.id),
        "filename": new_dataset.filename,
        "rows": new_dataset.rows,
        "columns": new_dataset.columns,
        "column_names": df.columns.tolist(),
        "memory_mb": new_dataset.memory_mb,
        "health_score": new_dataset.health_score,
        "eda": eda_data
    }

# --- Pydantic Schema ---
class TrainPayload(BaseModel):
    dataset_id: str
    target_column: str
    engine_mode: str
    features: Dict[str, bool]
    algos: Dict[str, bool]

# --- ENDPOINT 2: REAL ML ENGINE ---
@app.post("/api/train")
async def train_models(payload: TrainPayload):
    db = SessionLocal()
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    db.close()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found in database.")

    try:
        df = pd.read_csv(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {str(e)}")

    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)

    if payload.target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found in data.")
    
    df = df.dropna(subset=[payload.target_column])
    y = df[payload.target_column]
    X = df.drop(columns=[payload.target_column])

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
    X = X.fillna(0)

    if X.empty:
        raise HTTPException(status_code=400, detail="No usable numeric features found to train on.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_dictionary = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(eval_metric='logloss', random_state=42),
        "lightgbm": LGBMClassifier(random_state=42, verbose=-1),
        "svm": SVC(probability=True, random_state=42)
    }

    leaderboard = []
    
    print(f"\n--- 🚀 STARTING AUTOML PIPELINE (Rows: {len(X_train)} train, {len(X_test)} test) ---")
    
    for algo_key, is_selected in payload.algos.items():
        if is_selected and algo_key in model_dictionary:
            model = model_dictionary[algo_key]
            
            print(f"⏳ Training {algo_key.upper()}... ", end="", flush=True)
            
            start_time = time.time()
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                acc = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions, average='weighted')
                train_time = round(time.time() - start_time, 2)
                
                print(f"✅ DONE in {train_time}s (Acc: {acc:.4f})")
                
                # NEW: Serialize and save the model to the temp directory
                model_filename = f"{payload.dataset_id}_{algo_key}.joblib"
                model_filepath = os.path.join(TEMP_DIR, model_filename)
                joblib.dump(model, model_filepath)

                display_name = algo_key.replace("_", " ").title() if algo_key != "svm" else "SVM"
                if algo_key == "xgboost": display_name = "XGBoost"
                if algo_key == "lightgbm": display_name = "LightGBM"

                leaderboard.append({
                    "id": algo_key,
                    "name": display_name,
                    "accuracy": round(acc, 4),
                    "f1_score": round(f1, 4),
                    "train_time": train_time,
                    "download_url": f"http://localhost:8000/api/export/{model_filename}"
                })
            except Exception as e:
                print(f"❌ FAILED! Error: {str(e)}")
                continue

    print("--- 🏁 PIPELINE COMPLETE ---\n")

    if not leaderboard:
        raise HTTPException(status_code=500, detail="All selected algorithms failed to train.")

    leaderboard.sort(key=lambda x: x["accuracy"], reverse=True)
    
    return {
        "status": "success",
        "target_column": payload.target_column,
        "leaderboard": leaderboard
    }

# --- ENDPOINT 3: EXPORT MODEL ---
@app.get("/api/export/{filename}")
async def export_model(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found. It may have expired.")
    
    # Return the file as a downloadable attachment
    return FileResponse(
        path=file_path, 
        filename=filename, 
        media_type='application/octet-stream'
    )

@app.get("/")
def read_root():
    return {"message": "KyD.ai Backend is running."}