from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import numpy as np
import math
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
    score = 100 - (missing_pct * 150) - (duplicate_pct * 50)
    return max(0, int(score))

def safe_float(val):
    try:
        if pd.isna(val) or val == float('inf') or val == float('-inf'):
            return None
        return round(float(val), 2)
    except Exception:
        return None

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

    smart_insights = []
    if "SibSp" in df.columns and "Parch" in df.columns:
        smart_insights.append({"type": "feature", "message": "High predictive potential: Combine 'SibSp' and 'Parch' into a single 'FamilySize' feature."})
    
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() > 30 and df[col].nunique() < rows:
            smart_insights.append({"type": "warning", "message": f"High Cardinality: '{col}' has {df[col].nunique()} unique strings. Consider dropping it in the Setup Phase to prevent overfitting."})
        elif df[col].isnull().sum() / rows > 0.50:
            smart_insights.append({"type": "danger", "message": f"Data Sparsity: '{col}' is missing over 50% of its data. Imputation may introduce heavy bias. Consider dropping."})

    column_stats = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_count = int(df[col].isnull().sum())
        
        if pd.api.types.is_numeric_dtype(df[col]):
            clean_series = df[col].dropna()
            hist_data = []
            if not clean_series.empty:
                hist, bin_edges = np.histogram(clean_series, bins=10)
                hist_data = [{"name": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}", "count": int(hist[i])} for i in range(len(hist))]
            
            column_stats.append({
                "name": col, "type": "numeric", "dtype": dtype, "missing": missing_count,
                "mean": safe_float(df[col].mean()), "median": safe_float(df[col].median()),
                "min": safe_float(df[col].min()), "max": safe_float(df[col].max()),
                "std": safe_float(df[col].std()), "skewness": safe_float(df[col].skew()),
                "kurtosis": safe_float(df[col].kurt()), "chart_data": hist_data
            })
        else:
            top_counts = df[col].value_counts().head(5)
            bar_data = [{"name": str(k)[:10], "count": int(v)} for k, v in top_counts.items()]
            column_stats.append({
                "name": col, "type": "categorical", "dtype": dtype, "missing": missing_count,
                "unique": int(df[col].nunique()), "chart_data": bar_data
            })

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = {}
    if not numeric_df.empty:
        corr_df = numeric_df.corr().round(2)
        corr_matrix = {
            "columns": corr_df.columns.tolist(),
            "values": corr_df.fillna(0).values.tolist()
        }

    eda_data = {
        "missing_summary": missing_summary,
        "duplicate_count": int(df.duplicated().sum()),
        "column_stats": column_stats,
        "correlation_matrix": corr_matrix,
        "smart_insights": smart_insights 
    }

    db = SessionLocal()
    try:
        new_dataset = Dataset(
            id=file_id, filename=file.filename, file_path=file_path,
            rows=rows, columns=cols, memory_mb=round(memory_mb, 2),
            health_score=health_score, eda_json=eda_data
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
        "dataset_id": str(new_dataset.id), "filename": new_dataset.filename,
        "rows": new_dataset.rows, "columns": new_dataset.columns,
        "column_names": df.columns.tolist(), "memory_mb": new_dataset.memory_mb,
        "health_score": new_dataset.health_score, "eda": eda_data
    }

class BivariatePayload(BaseModel):
    dataset_id: str
    target_column: str

@app.post("/api/bivariate")
async def get_bivariate(payload: BivariatePayload):
    db = SessionLocal()
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    db.close()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = pd.read_csv(dataset.file_path)
    
    if payload.target_column not in df.columns or df[payload.target_column].nunique() > 10:
        return {"status": "skipped", "message": "Bivariate stacking currently optimized for classification targets (<=10 classes)."}

    target_cats = [str(x) for x in df[payload.target_column].dropna().unique()]
    df['__target__'] = df[payload.target_column].astype(str)
    
    bivariate_results = {}
    
    for col in df.columns:
        if col == payload.target_column or col == '__target__': continue
        chart_data = []
        
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            clean_series = df[col].dropna()
            if clean_series.empty: continue
            bins = np.histogram_bin_edges(clean_series, bins=10)
            df['__temp_bin__'] = pd.cut(df[col], bins=bins, include_lowest=True).astype(str)
            cross = pd.crosstab(df['__temp_bin__'], df['__target__'])
            
            for i in range(len(bins)-1):
                bin_label = f"({bins[i]:.3g}, {bins[i+1]:.3g}]"
                matching_idx = [idx for idx in cross.index if str(round(bins[i], 1)) in idx or str(round(bins[i+1], 1)) in idx]
                
                entry = {"name": f"{bins[i]:.1f}-{bins[i+1]:.1f}"}
                for tc in target_cats:
                    val = 0
                    if matching_idx and tc in cross.columns:
                        val = int(cross.loc[matching_idx[0], tc])
                    entry[tc] = val
                chart_data.append(entry)
            df.drop(columns=['__temp_bin__'], inplace=True)
            
        else:
            top_cats = df[col].value_counts().head(8).index
            filtered_df = df[df[col].isin(top_cats)]
            cross = pd.crosstab(filtered_df[col], filtered_df['__target__'])
            for cat_val in cross.index:
                entry = {"name": str(cat_val)[:15]}
                for tc in target_cats:
                    entry[tc] = int(cross.loc[cat_val, tc]) if tc in cross.columns else 0
                chart_data.append(entry)
                
        bivariate_results[col] = chart_data

    return {"status": "success", "target_classes": target_cats, "data": bivariate_results}

# --- UPGRADE: Pydantic Schema now accepts preprocessing instructions ---
class TrainPayload(BaseModel):
    dataset_id: str
    target_column: str
    engine_mode: str
    features: Dict[str, bool]
    algos: Dict[str, bool]
    drop_columns: List[str] = [] # NEW
    imputation_strategy: Dict[str, str] = {} # NEW

# --- ENDPOINT 2: REAL ML ENGINE WITH PREPROCESSING ---
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
    
    print("\n--- 🔧 EXECUTING PREPROCESSING PIPELINE ---")
    
    # 1. DROP COLUMNS REQUESTED BY USER
    cols_to_drop = [col for col in payload.drop_columns if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")

    # 2. APPLY IMPUTATION STRATEGIES FROM EDA HUB
    for col, strategy in payload.imputation_strategy.items():
        if col in df.columns and col != payload.target_column:
            if strategy == 'drop_col':
                df = df.drop(columns=[col])
                print(f"Imputation: Dropped column '{col}'")
            elif strategy == 'drop_rows':
                df = df.dropna(subset=[col])
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'mode':
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df[col] = df[col].fillna("Unknown")
            elif strategy == 'ffill':
                df[col] = df[col].fillna(method='ffill')
            elif strategy == 'bfill':
                df[col] = df[col].fillna(method='bfill')

    # Drop target column NaNs
    df = df.dropna(subset=[payload.target_column])
    y = df[payload.target_column]
    X = df.drop(columns=[payload.target_column])

    # Encode Target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # For now, drop remaining raw text columns to prevent algorithm crashes
    X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
    X = X.fillna(0)

    if X.empty:
        raise HTTPException(status_code=400, detail="No usable numeric features found to train on after preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_dictionary = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(eval_metric='logloss', random_state=42),
        "lightgbm": LGBMClassifier(random_state=42, verbose=-1),
        "svm": SVC(probability=True, random_state=42)
    }

    leaderboard = []
    
    print(f"--- 🚀 STARTING AUTOML PIPELINE (Rows: {len(X_train)} train, {len(X_test)} test) ---")
    
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

@app.get("/api/export/{filename}")
async def export_model(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found. It may have expired.")
    
    return FileResponse(
        path=file_path, 
        filename=filename, 
        media_type='application/octet-stream'
    )

@app.get("/")
def read_root():
    return {"message": "KyD.ai Backend is running."}