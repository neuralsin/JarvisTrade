"""
Portfolio bucket tracking and management
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from typing import List, Optional
from app.db.database import get_db
from app.db.models import User
from app.routers.auth import get_current_user
from uuid import UUID, uuid4
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class BucketCreate(BaseModel):
    name: str
    bucket_type: str  # 'sector', 'market_cap', 'strategy', 'custom'
    target_allocation: float  # 0.0 to 100.0
    rules: Optional[dict] = {}


class BucketResponse(BaseModel):
    id: str
    name: str
    bucket_type: str
    target_allocation: float
    current_allocation: float
    current_value: float
    rules: dict
    positions_count: int


class AllocationResponse(BaseModel):
    bucket_name: str
    target: float
    current: float
    difference: float
    value: float


# Since we haven't created the Bucket model yet, we'll use SystemState for simple storage
# In production, you'd add proper Bucket and BucketPosition models to models.py

@router.get("/")
async def get_buckets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all buckets for current user
    """
    # For now, store in SystemState as JSON
    # In production, query from buckets table
    from app.db.models import SystemState
    import json
    
    bucket_key = f"buckets_{current_user.id}"
    bucket_record = db.query(SystemState).filter(SystemState.key == bucket_key).first()
    
    if not bucket_record or not bucket_record.value:
        # Return default buckets
        return {
            "buckets": [
                {
                    "id": str(uuid4()),
                    "name": "IT Sector",
                    "bucket_type": "sector",
                    "target_allocation": 40.0,
                    "current_allocation": 0.0,
                    "current_value": 0.0,
                    "rules": {"sector": "IT"},
                    "positions_count": 0
                },
                {
                    "id": str(uuid4()),
                    "name": "Large Cap",
                    "bucket_type": "market_cap",
                    "target_allocation": 60.0,
                    "current_allocation": 0.0,
                    "current_value": 0.0,
                    "rules": {"market_cap": "large"},
                    "positions_count": 0
                }
            ]
        }
    
    buckets = json.loads(bucket_record.value)
    
    # Calculate current allocations based on open trades
    from app.db.models import Trade
    
    open_trades = db.query(Trade).filter(
        Trade.user_id == current_user.id,
        Trade.status == 'open'
    ).all()
    
    # Simple allocation: count trades per bucket
    for bucket in buckets:
        matching_trades = [t for t in open_trades if _matches_bucket_rules(t, bucket['rules'])]
        bucket['positions_count'] = len(matching_trades)
        bucket['current_value'] = sum([(t.entry_price * t.qty) for t in matching_trades])
    
    total_value = sum([b['current_value'] for b in buckets]) or 1.0
    for bucket in buckets:
        bucket['current_allocation'] = (bucket['current_value'] / total_value) * 100
    
    return {"buckets": buckets}


def _matches_bucket_rules(trade, rules):
    """Check if trade matches bucket rules"""
    # Simplified - in production, check against instrument metadata
    return True  # Placeholder


@router.post("/")
async def create_bucket(
    bucket: BucketCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new bucket
    """
    from app.db.models import SystemState
    import json
    
    bucket_key = f"buckets_{current_user.id}"
    bucket_record = db.query(SystemState).filter(SystemState.key == bucket_key).first()
    
    if bucket_record and bucket_record.value:
        buckets = json.loads(bucket_record.value)
    else:
        buckets = []
    
    new_bucket = {
        "id": str(uuid4()),
        "name": bucket.name,
        "bucket_type": bucket.bucket_type,
        "target_allocation": bucket.target_allocation,
        "current_allocation": 0.0,
        "current_value": 0.0,
        "rules": bucket.rules,
        "positions_count": 0
    }
    
    buckets.append(new_bucket)
    
    if bucket_record:
        bucket_record.value = json.dumps(buckets)
    else:
        bucket_record = SystemState(key=bucket_key, value=json.dumps(buckets))
        db.add(bucket_record)
    
    db.commit()
    
    logger.info(f"Created bucket: {bucket.name} for user {current_user.email}")
    
    return {"status": "success", "bucket": new_bucket}


@router.get("/allocation")
async def get_allocation_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get portfolio allocation across all buckets
    """
    buckets_response = await get_buckets(current_user, db)
    buckets = buckets_response['buckets']
    
    allocations = []
    for bucket in buckets:
        allocations.append({
            "bucket_name": bucket['name'],
            "target": bucket['target_allocation'],
            "current": bucket['current_allocation'],
            "difference": bucket['current_allocation'] - bucket['target_allocation'],
            "value": bucket['current_value']
        })
    
    return {
        "allocations": allocations,
        "total_value": sum([a['value'] for a in allocations]),
        "rebalance_needed": any([abs(a['difference']) > 5.0 for a in allocations])
    }


@router.get("/rebalance")
async def get_rebalance_suggestions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Suggest trades to rebalance portfolio to target allocation
    """
    allocation = await get_allocation_summary(current_user, db)
    
    suggestions = []
    for alloc in allocation['allocations']:
        diff = alloc['difference']
        if abs(diff) > 5.0:  # If more than 5% off target
            action = "reduce" if diff > 0 else "increase"
            amount = abs(diff) * allocation['total_value'] / 100
            
            suggestions.append({
                "bucket": alloc['bucket_name'],
                "action": action,
                "amount": amount,
                "reason": f"Currently {diff:+.1f}% from target"
            })
    
    return {
        "suggestions": suggestions,
        "priority": "high" if len(suggestions) > 0 else "none"
    }


@router.delete("/{bucket_id}")
async def delete_bucket(
    bucket_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a bucket
    """
    from app.db.models import SystemState
    import json
    
    bucket_key = f"buckets_{current_user.id}"
    bucket_record = db.query(SystemState).filter(SystemState.key == bucket_key).first()
    
    if not bucket_record or not bucket_record.value:
        raise HTTPException(status_code=404, detail="No buckets found")
    
    buckets = json.loads(bucket_record.value)
    buckets = [b for b in buckets if b['id'] != bucket_id]
    
    bucket_record.value = json.dumps(buckets)
    db.commit()
    
    return {"status": "success", "message": "Bucket deleted"}
