# models/result_schema.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


class ErrorResponse(BaseModel):
    """错误响应模型（解决递归引用问题）[9,10](@ref)"""
    error: str = Field(..., examples=["Processing Failed"], min_length=3)
    details: Optional[str] = Field(None, examples=["Invalid EEG format"])

    model_config = ConfigDict(
        json_schema_extra={"example": {"error": "Validation Error", "details": "Field 'sfreq' required"}}
    )


class ProcessingResult(BaseModel):
    """处理结果模型（修复类型不匹配错误）[10](@ref)"""
    status: str = Field(..., pattern=r"^(success|failed|pending)$")
    duration: float = Field(..., gt=0, le=300)
    artifact_count: int = Field(0, ge=0)
    metrics: dict = Field(default_factory=dict)

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={"example": {"status": "success", "duration": 5.2, "artifact_count": 3}}
    )