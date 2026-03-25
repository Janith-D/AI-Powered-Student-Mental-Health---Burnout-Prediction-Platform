"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import json


class StudentDataInput(BaseModel):
    """Student data for prediction"""
    sleep_quality: float = Field(..., ge=1, le=10, description="Sleep quality rating (1-10)")
    stress_level: float = Field(..., ge=0, le=100, description="Stress level (0-100)")
    exercise_hours: float = Field(..., ge=0, le=24, description="Hours of exercise per week")
    anxiety_score: float = Field(..., ge=0, le=100, description="Anxiety score (0-100)")
    social_support: float = Field(..., ge=1, le=10, description="Social support quality (1-10)")
    meditation_hours: float = Field(..., ge=0, le=10, description="Hours of meditation per week")
    depression_score: float = Field(..., ge=0, le=100, description="Depression score (0-100)")
    study_hours: float = Field(..., ge=0, le=24, description="Hours studying per day")
    course_difficulty: float = Field(..., ge=1, le=10, description="Course difficulty (1-10)")
    assignment_load: float = Field(..., ge=1, le=10, description="Assignment load (1-10)")
    gender: Optional[str] = Field(default="Male", description="Gender (Male/Female/Other)")
    major: Optional[str] = Field(default="STEM", description="Major (STEM/Business/Humanities/Arts)")

    class Config:
        json_schema_extra = {
            "example": {
                "sleep_quality": 7.5,
                "stress_level": 45,
                "exercise_hours": 3,
                "anxiety_score": 30,
                "social_support": 8,
                "meditation_hours": 1,
                "depression_score": 20,
                "study_hours": 6,
                "course_difficulty": 7,
                "assignment_load": 6,
                "gender": "Female",
                "major": "STEM"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output"""
    burnout_score: float = Field(..., description="Predicted burnout score (0-100)")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    confidence_burnout: float = Field(..., description="Model confidence (0-1)")
    message: str = Field(..., description="Prediction summary")


class FeatureContribution(BaseModel):
    """Single feature contribution to SHAP explanation"""
    feature_name: str
    feature_value: float
    shap_value: float
    direction: str  # "increases" or "decreases"
    impact: str  # "strong", "moderate", "weak"


class ExplanationResponse(BaseModel):
    """SHAP explanation output"""
    burnout_score: float
    base_value: float
    prediction_difference: float
    top_contributors: List[FeatureContribution]
    protection_factors: List[str]
    risk_factors: List[str]
    summary: str


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    regression_model: str
    classification_model: str
    feature_count: int


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    status_code: int
