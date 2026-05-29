from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import pandas as pd
import os
import uuid
from models import Base, Dataset

# Load environment variables
load_dotenv()

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing.")

# Convert asyncpg string to psycopg2 for synchronous operations initially
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# UPDATED: Robust connection pooling for serverless Neon
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # Pings the DB to check if the connection is alive
    pool_recycle=300,         # Preemptively recycles connections every 5 minutes
    pool_size=5,              # Keeps the baseline pool small
    max_overflow=10,          # Allows a few extra connections during traffic spikes
    connect_args={
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# FastAPI Setup
app = FastAPI(title="KyD.ai API")

# Allow the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Your Vite frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp directory exists
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

def calculate_health_score(df: pd.DataFrame) -> int:
    # A simplified health score for now
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    duplicate_pct = df.duplicated().sum() / df.shape[0]
    
    score = 100 - (missing_pct * 40) - (duplicate_pct * 20)
    return max(0, int(score))

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, f"{file_id}_{file.filename}")

    # Save the file temporarily
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Load into pandas for basic stats
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

    # Calculate basic EDA
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

    # Save to Database
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

    # Return the exact payload expected by the frontend
    return {
        "dataset_id": str(new_dataset.id),
        "filename": new_dataset.filename,
        "rows": new_dataset.rows,
        "columns": new_dataset.columns,
        "column_names": df.columns.tolist(), # <--- ADD THIS LINE
        "memory_mb": new_dataset.memory_mb,
        "health_score": new_dataset.health_score,
        "eda": eda_data
    }

@app.get("/")
def read_root():
    return {"message": "KyD.ai Backend is running."}