"""
CSV Upload Router for Model Training
Allows users to upload CSV files with OHLCV data for training models
"""
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from app.db.database import get_db
from app.db.models import User, Instrument
from app.routers.auth import get_current_user
from datetime import datetime
import pandas as pd
import os
import uuid
import logging
import shutil

logger = logging.getLogger(__name__)
router = APIRouter()

# Directory for storing uploaded CSV files
UPLOAD_DIR = "uploads/csv"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class CSVDatasetInfo(BaseModel):
    id: str
    filename: str
    symbol: str
    row_count: int
    date_start: str
    date_end: str
    uploaded_at: str
    columns: List[str]


class CSVValidationResult(BaseModel):
    valid: bool
    error: Optional[str] = None
    row_count: int = 0
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    columns: List[str] = []
    preview: List[dict] = []


def validate_csv(df: pd.DataFrame) -> CSVValidationResult:
    """Validate that CSV has required columns for OHLCV data"""
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    missing = required_cols - set(df.columns)
    if missing:
        return CSVValidationResult(
            valid=False,
            error=f"Missing required columns: {', '.join(missing)}. Need: open, high, low, close, volume",
            columns=list(df.columns)
        )
    
    # Check for date column
    date_col = None
    for col in ['date', 'datetime', 'timestamp', 'time', 'ts', 'ts_utc']:
        if col in df.columns:
            date_col = col
            break
    
    if not date_col:
        # Try to use index if it's a DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'date'})
            date_col = 'date'
        else:
            return CSVValidationResult(
                valid=False,
                error="No date column found. Need one of: date, datetime, timestamp, time, ts, ts_utc",
                columns=list(df.columns)
            )
    
    # Parse dates
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
    except Exception as e:
        return CSVValidationResult(
            valid=False,
            error=f"Failed to parse dates: {str(e)}",
            columns=list(df.columns)
        )
    
    # Get date range
    date_start = df[date_col].min().strftime('%Y-%m-%d')
    date_end = df[date_col].max().strftime('%Y-%m-%d')
    
    # Preview first 5 rows
    preview = df.head(5).to_dict('records')
    for row in preview:
        for k, v in row.items():
            if pd.isna(v):
                row[k] = None
            elif isinstance(v, pd.Timestamp):
                row[k] = v.isoformat()
    
    return CSVValidationResult(
        valid=True,
        row_count=len(df),
        date_start=date_start,
        date_end=date_end,
        columns=list(df.columns),
        preview=preview
    )


@router.post("/validate")
async def validate_csv_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Validate a CSV file before upload
    Returns validation result with preview
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse CSV
        from io import StringIO
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        # Validate
        result = validate_csv(df)
        
        return result
    
    except Exception as e:
        logger.error(f"CSV validation failed: {e}")
        return CSVValidationResult(
            valid=False,
            error=f"Failed to parse CSV: {str(e)}",
            columns=[]
        )


@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    symbol: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a CSV file for model training
    
    Args:
        file: CSV file with OHLCV data
        symbol: Stock symbol (e.g., RELIANCE)
    
    Returns:
        Dataset info with ID for training
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read and validate
        content = await file.read()
        from io import StringIO
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        validation = validate_csv(df)
        if not validation.valid:
            raise HTTPException(status_code=400, detail=validation.error)
        
        # Generate unique ID
        dataset_id = str(uuid.uuid4())
        
        # Save file
        file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_{symbol.upper()}.csv")
        
        # Reset file position and save
        await file.seek(0)
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Ensure instrument exists
        symbol_upper = symbol.upper().strip()
        instrument = db.query(Instrument).filter(Instrument.symbol == symbol_upper).first()
        if not instrument:
            # Create instrument
            instrument = Instrument(
                symbol=symbol_upper,
                name=symbol_upper,
                exchange='CSV',
                instrument_type='EQ'
            )
            db.add(instrument)
            db.commit()
            logger.info(f"Created instrument {symbol_upper} from CSV upload")
        
        logger.info(f"CSV uploaded: {file.filename} -> {file_path}")
        
        return CSVDatasetInfo(
            id=dataset_id,
            filename=file.filename,
            symbol=symbol_upper,
            row_count=validation.row_count,
            date_start=validation.date_start,
            date_end=validation.date_end,
            uploaded_at=datetime.utcnow().isoformat(),
            columns=validation.columns
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/datasets")
async def list_datasets(
    current_user: User = Depends(get_current_user)
):
    """
    List all uploaded CSV datasets
    """
    datasets = []
    
    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            if filename.endswith('.csv'):
                file_path = os.path.join(UPLOAD_DIR, filename)
                
                # Parse ID and symbol from filename
                parts = filename.replace('.csv', '').split('_', 1)
                if len(parts) == 2:
                    dataset_id = parts[0]
                    symbol = parts[1]
                else:
                    dataset_id = filename.replace('.csv', '')
                    symbol = 'UNKNOWN'
                
                # Get file info
                try:
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.lower()
                    
                    # Find date column
                    date_col = None
                    for col in ['date', 'datetime', 'timestamp', 'time', 'ts', 'ts_utc']:
                        if col in df.columns:
                            date_col = col
                            break
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col])
                        date_start = df[date_col].min().strftime('%Y-%m-%d')
                        date_end = df[date_col].max().strftime('%Y-%m-%d')
                    else:
                        date_start = 'Unknown'
                        date_end = 'Unknown'
                    
                    stat = os.stat(file_path)
                    
                    datasets.append({
                        'id': dataset_id,
                        'filename': filename,
                        'symbol': symbol,
                        'row_count': len(df),
                        'date_start': date_start,
                        'date_end': date_end,
                        'uploaded_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'file_size': stat.st_size
                    })
                except Exception as e:
                    logger.warning(f"Failed to read dataset {filename}: {e}")
    
    return {'datasets': datasets}


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete an uploaded CSV dataset
    """
    # Find file by ID
    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(dataset_id):
                file_path = os.path.join(UPLOAD_DIR, filename)
                os.remove(file_path)
                logger.info(f"Deleted dataset: {filename}")
                return {'status': 'deleted', 'id': dataset_id}
    
    raise HTTPException(status_code=404, detail="Dataset not found")


def load_csv_for_training(dataset_id: str, symbol: str = None) -> pd.DataFrame:
    """
    Load a CSV dataset for training
    Called by model_training.py
    
    Returns DataFrame with columns: ts_utc, open, high, low, close, volume
    """
    # Find file
    file_path = None
    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(dataset_id):
                file_path = os.path.join(UPLOAD_DIR, filename)
                break
    
    if not file_path:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    # Load CSV
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.strip()
    
    # Find and rename date column to ts_utc
    date_col = None
    for col in ['date', 'datetime', 'timestamp', 'time', 'ts', 'ts_utc']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col and date_col != 'ts_utc':
        df = df.rename(columns={date_col: 'ts_utc'})
    
    df['ts_utc'] = pd.to_datetime(df['ts_utc'])
    
    # Ensure required columns
    required = ['ts_utc', 'open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Sort by date
    df = df.sort_values('ts_utc').reset_index(drop=True)
    
    logger.info(f"Loaded CSV dataset {dataset_id}: {len(df)} rows")
    
    return df[required]
