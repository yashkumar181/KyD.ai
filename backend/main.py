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
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from models import Base, Dataset

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing.")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, pool_pre_ping=True, pool_recycle=300,         
    pool_size=5, max_overflow=10,          
    connect_args={"keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 5}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="KyD.ai API")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173"], 
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
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
            smart_insights.append({"type": "warning", "message": f"High Cardinality: '{col}' has {df[col].nunique()} unique strings. Consider dropping it in Setup."})
        elif df[col].isnull().sum() / rows > 0.50:
            smart_insights.append({"type": "danger", "message": f"Data Sparsity: '{col}' is missing over 50% of its data. Consider dropping."})

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
        corr_matrix = {"columns": numeric_df.corr().columns.tolist(), "values": numeric_df.corr().round(2).fillna(0).values.tolist()}

    eda_data = {
        "missing_summary": missing_summary, "duplicate_count": int(df.duplicated().sum()),
        "column_stats": column_stats, "correlation_matrix": corr_matrix, "smart_insights": smart_insights 
    }

    # EXPORT GENERATION: Create Markdown EDA Report
    eda_md = f"# KyD.ai Data Intelligence Report\n\n## Dataset Overview\n- **File:** {file.filename}\n- **Rows:** {rows}\n- **Columns:** {cols}\n- **Health Score:** {health_score}/100\n\n"
    eda_md += "## Missing Data Issues\n"
    for m in missing_summary: eda_md += f"- **{m['column']}**: {m['missing_pct']}% missing\n"
    eda_md += "\n## AI Smart Insights\n"
    for s in smart_insights: eda_md += f"- **[{s['type'].upper()}]**: {s['message']}\n"
    
    eda_filename = f"{file_id}_eda_report.md"
    with open(os.path.join(TEMP_DIR, eda_filename), "w", encoding="utf-8") as f:
        f.write(eda_md)

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
        "health_score": new_dataset.health_score, "eda": eda_data,
        "eda_download_url": f"http://localhost:8000/api/export/{eda_filename}" # NEW EXPORT URL
    }

class BivariatePayload(BaseModel):
    dataset_id: str
    target_column: str

@app.post("/api/bivariate")
async def get_bivariate(payload: BivariatePayload):
    db = SessionLocal()
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    db.close()
    if not dataset: raise HTTPException(status_code=404, detail="Dataset not found.")

    df = pd.read_csv(dataset.file_path)
    if payload.target_column not in df.columns or df[payload.target_column].nunique() > 10:
        return {"status": "skipped", "message": "Bivariate stacking currently optimized for classification targets."}

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
                matching_idx = [idx for idx in cross.index if str(round(bins[i], 1)) in idx or str(round(bins[i+1], 1)) in idx]
                entry = {"name": f"{bins[i]:.1f}-{bins[i+1]:.1f}"}
                for tc in target_cats:
                    entry[tc] = int(cross.loc[matching_idx[0], tc]) if matching_idx and tc in cross.columns else 0
                chart_data.append(entry)
            df.drop(columns=['__temp_bin__'], inplace=True)
        else:
            top_cats = df[col].value_counts().head(8).index
            cross = pd.crosstab(df[df[col].isin(top_cats)][col], df[df[col].isin(top_cats)]['__target__'])
            for cat_val in cross.index:
                entry = {"name": str(cat_val)[:15]}
                for tc in target_cats: entry[tc] = int(cross.loc[cat_val, tc]) if tc in cross.columns else 0
                chart_data.append(entry)
        bivariate_results[col] = chart_data

    return {"status": "success", "target_classes": target_cats, "data": bivariate_results}

class TrainPayload(BaseModel):
    dataset_id: str
    target_column: str
    engine_mode: str
    features: Dict[str, bool]
    algos: Dict[str, bool]
    drop_columns: List[str] = [] 
    imputation_strategy: Dict[str, str] = {} 

@app.post("/api/train")
async def train_models(payload: TrainPayload):
    db = SessionLocal()
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    db.close()
    
    if not dataset: raise HTTPException(status_code=404, detail="Dataset not found in database.")

    try: df = pd.read_csv(dataset.file_path)
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to read CSV: {str(e)}")

    if len(df) > 20000: df = df.sample(n=20000, random_state=42)
    if payload.target_column not in df.columns: raise HTTPException(status_code=400, detail="Target column not found.")
    
    print("\n--- 🔧 EXECUTING PREPROCESSING PIPELINE ---")
    cols_to_drop = [col for col in payload.drop_columns if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    for col, strategy in payload.imputation_strategy.items():
        if col in df.columns and col != payload.target_column:
            if strategy == 'drop_col': df = df.drop(columns=[col])
            elif strategy == 'drop_rows': df = df.dropna(subset=[col])
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]): df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]): df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'mode': 
                if not df[col].mode().empty: df[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant': df[col] = df[col].fillna("Unknown")

    df = df.dropna(subset=[payload.target_column])
    y = df[payload.target_column]
    X = df.drop(columns=[payload.target_column])

    le = LabelEncoder()
    y = le.fit_transform(y)
    X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).fillna(0)

    if X.empty: raise HTTPException(status_code=400, detail="No usable numeric features found to train on.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_dictionary = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(eval_metric='logloss', random_state=42),
        "lightgbm": LGBMClassifier(random_state=42, verbose=-1),
        "svm": SVC(probability=True, random_state=42)
    }

    leaderboard = []
    
    for algo_key, is_selected in payload.algos.items():
        if is_selected and algo_key in model_dictionary:
            model = model_dictionary[algo_key]
            start_time = time.time()
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                acc = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions, average='weighted')
                train_time = round(time.time() - start_time, 2)
                
                model_filename = f"{payload.dataset_id}_{algo_key}.joblib"
                joblib.dump(model, os.path.join(TEMP_DIR, model_filename))

                display_name = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "svm": "SVM"}.get(algo_key, algo_key.replace("_", " ").title())

                leaderboard.append({
                    "id": algo_key, "name": display_name,
                    "accuracy": round(acc, 4), "f1_score": round(f1, 4), "train_time": train_time,
                    "download_url": f"http://localhost:8000/api/export/{model_filename}"
                })
            except Exception as e:
                print(f"❌ FAILED! Error: {str(e)}")
                continue

    if not leaderboard: raise HTTPException(status_code=500, detail="All selected algorithms failed to train.")
    leaderboard.sort(key=lambda x: x["accuracy"], reverse=True)
    
    # EXPORT GENERATION: Create Jupyter Notebook (.ipynb)
    notebook_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [f"# KyD.ai - Automated ML Pipeline\n", f"### Target: `{payload.target_column}` | Best Model: `{leaderboard[0]['name']}`\n", f"Auto-generated code achieving **{(leaderboard[0]['accuracy']*100):.2f}%** accuracy."]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["import pandas as pd\n", "from sklearn.model_selection import train_test_split\n", "import joblib\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [f"# 1. Load Data\n", f"df = pd.read_csv('your_dataset.csv')\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [f"# 2. Preprocessing\n", f"cols_to_drop = {cols_to_drop}\n", f"df = df.drop(columns=cols_to_drop, errors='ignore')\n", "# (Add your imputation logic here based on EDA choices)\n", f"df = df.dropna(subset=['{payload.target_column}'])\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [f"# 3. Train/Test Split\n", f"X = df.select_dtypes(include=['number']).drop(columns=['{payload.target_column}'])\n", f"y = df['{payload.target_column}']\n", "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [f"# 4. Load & Predict using {leaderboard[0]['name']}\n", f"model = joblib.load('{payload.dataset_id}_{leaderboard[0]['id']}.joblib')\n", f"predictions = model.predict(X_test)\n", f"print(predictions)\n"]}
    ]
    nb_filename = f"{payload.dataset_id}_pipeline.ipynb"
    with open(os.path.join(TEMP_DIR, nb_filename), "w") as f:
        json.dump({"cells": notebook_cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, f)

    return {
        "status": "success", "target_column": payload.target_column, "leaderboard": leaderboard,
        "notebook_download_url": f"http://localhost:8000/api/export/{nb_filename}" # NEW EXPORT URL
    }

@app.get("/api/export/{filename}")
async def export_model(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path): raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

@app.get("/")
def read_root(): return {"message": "KyD.ai Backend is running."}