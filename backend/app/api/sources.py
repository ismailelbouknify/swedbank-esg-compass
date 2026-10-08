"""Source details + safe file access for stored PDFs."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.serializers import source_out
from app.core.config import get_settings
from app.db.session import get_db
from app.models.entities import Evidence, Source
from app.schemas.api import SourceDetail

router = APIRouter(prefix="/api/sources", tags=["sources"])


def _get(db: Session, source_id: int) -> Source:
    s = db.get(Source, source_id)
    if s is None:
        raise HTTPException(404, "Source not found")
    return s


@router.get("/{source_id}", response_model=SourceDetail)
def get_source(source_id: int, db: Session = Depends(get_db)) -> SourceDetail:
    s = _get(db, source_id)
    count = db.scalar(select(func.count(Evidence.id)).where(Evidence.source_id == s.id)) or 0
    return SourceDetail(**source_out(s).model_dump(), assessment_id=s.assessment_id, evidence_count=count)


@router.get("/{source_id}/file")
def get_source_file(source_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """Serve a stored PDF. Only files inside the data directory are ever served."""
    s = _get(db, source_id)
    if not s.file_path:
        raise HTTPException(404, "No stored file for this source")
    data_dir = Path(get_settings().data_dir).resolve()
    path = Path(s.file_path).resolve()
    if not path.is_relative_to(data_dir) or not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(404, "File not available")
    return FileResponse(path, media_type="application/pdf", filename=s.original_filename or path.name,
                        content_disposition_type="inline")
