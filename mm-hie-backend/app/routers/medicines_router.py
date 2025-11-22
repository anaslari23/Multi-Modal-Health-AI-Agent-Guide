from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional

from app.database import get_db
from app.db_models import Medicine

router = APIRouter(prefix="/medicines", tags=["medicines"])

@router.get("/search", response_model=List[dict])
def search_medicines(
    q: Optional[str] = Query(None, description="Search query for medicine name"),
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Search for medicines by name. If query is empty, returns a default list.
    """
    query = db.query(Medicine)
    
    if q and len(q.strip()) > 0:
        query = query.filter(Medicine.name.ilike(f"%{q}%"))
        
    medicines = query.limit(limit).all()
    
    return [
        {
            "id": m.id,
            "name": m.name,
            "price": m.price,
            "manufacturer_name": m.manufacturer_name,
            "type": m.type,
            "pack_size_label": m.pack_size_label,
            "short_composition1": m.short_composition1,
            "short_composition2": m.short_composition2,
            "medicine_desc": m.medicine_desc,
            "side_effects": m.side_effects,
            "drug_interactions": m.drug_interactions,
        }
        for m in medicines
    ]
