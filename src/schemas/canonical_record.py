from pydantic import BaseModel
from typing import Optional, Dict


class CanonicalRecord(BaseModel):

    timestamp: str
    geo_id: str
    geo_sub: Optional[str] = None

    source: str
    entity_type: str

    value: float
    unit: Optional[str] = None

    data_status: str = "observed"

    pull_ts: str

    metadata: Dict