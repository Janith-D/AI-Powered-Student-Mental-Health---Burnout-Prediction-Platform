"""
FastAPI application for Student Mental Health Burnout Prediction
Production-ready API with prediction and explainability endpoints
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn

from schemas import StudentDataInput, PredictionResponse, ExplanationResponse, HealthCheckResponse, ErrorResponse, FeatureContribution
from model_loader import model_loader
from shap_api_utils import prepare_data_for_prediction, get_shap_explanation, validate_data


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting FastAPI application...")
    try:
        model_loader.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application...")


# Create FastAPI app
app = FastAPI(
    title="Burnout Prediction API",
    description="Student Mental Health & Burnout Score Prediction with SHAP Explanations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check API health and model status"""
    try:
        return HealthCheckResponse(
            status="healthy" if model_loader.is_loaded else "unhealthy",
            models_loaded=model_loader.is_loaded,
            regression_model="XGBoost (200 trees, depth=6)",
            classification_model="XGBoost Classifier (200 trees, depth=6)",
            feature_count=len(model_loader.features) if model_loader.features else 0
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(student_data: StudentDataInput):
    """
    Predict burnout score and risk level for a student

    **Input:**
    - Student mental health and academic data

    **Output:**
    - burnout_score: Predicted burnout (0-100)
    - risk_level: Risk category (Low/Medium/High)
    - confidence_burnout: Model confidence (0-1)
    """
    try:
        # Validate input
        is_valid, message = validate_data(student_data)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=message
            )

        # Check models loaded
        if not model_loader.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded"
            )

        logger.info("Processing prediction request...")

        # Prepare data
        X_scaled, X_original = prepare_data_for_prediction(
            student_data,
            model_loader.features,
            model_loader.scaler
        )

        # Make predictions
        predictions = model_loader.predict(X_scaled)

        # Prepare response
        burnout_score = predictions['burnout_score']
        risk_level = predictions['risk_level']
        confidence = predictions['confidence']

        # Generate message
        if risk_level == "High":
            message = f"Student has HIGH burnout risk (score: {burnout_score:.1f}/100). Intervention recommended."
        elif risk_level == "Medium":
            message = f"Student has MEDIUM burnout risk (score: {burnout_score:.1f}/100). Monitor closely."
        else:
            message = f"Student has LOW burnout risk (score: {burnout_score:.1f}/100). Keep up good practices."

        logger.info(f"Prediction successful: {risk_level} risk, burnout={burnout_score:.2f}")

        return PredictionResponse(
            burnout_score=burnout_score,
            risk_level=risk_level,
            confidence_burnout=confidence,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse)
async def explain(student_data: StudentDataInput):
    """
    Predict and explain using SHAP for a student

    **Input:**
    - Student mental health and academic data

    **Output:**
    - burnout_score: Predicted burnout
    - top_contributors: Features impacting prediction most
    - protection_factors: Factors protecting against burnout
    - risk_factors: Factors increasing burnout
    - summary: Human-readable explanation
    """
    try:
        # Validate input
        is_valid, message = validate_data(student_data)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=message
            )

        # Check models loaded
        if not model_loader.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded"
            )

        logger.info("Processing explanation request...")

        # Prepare data
        X_scaled, X_original = prepare_data_for_prediction(
            student_data,
            model_loader.features,
            model_loader.scaler
        )

        # Get SHAP explainer
        explainer = model_loader.get_shap_explainer()
        if explainer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SHAP explainer not available. Install SHAP: pip install shap"
            )

        # Get explanation
        explanation_data = get_shap_explanation(
            explainer,
            X_scaled,
            X_original,
            model_loader.features,
            sample_idx=0
        )

        if explanation_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate explanation")

        # Convert to response format
        contributors = [
            FeatureContribution(
                feature_name=c['feature_name'],
                feature_value=c['feature_value'],
                shap_value=c['shap_value'],
                direction=c['direction'],
                impact=c['impact']
            )
            for c in explanation_data['top_contributors']
        ]

        logger.info(f"Explanation generated successfully for burnout={explanation_data['burnout_score']:.2f}")

        return ExplanationResponse(
            burnout_score=explanation_data['burnout_score'],
            base_value=explanation_data['base_value'],
            prediction_difference=explanation_data['prediction_difference'],
            top_contributors=contributors,
            protection_factors=explanation_data['protection_factors'],
            risk_factors=explanation_data['risk_factors'],
            summary=explanation_data['summary']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API documentation"""
    return {
        "name": "Student Mental Health Burnout Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "explain": "POST /explain",
            "docs": "GET /docs"
        },
        "documentation": "/docs"
    }


@app.get("/features")
async def get_features():
    """Get list of model features"""
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {
        "feature_count": len(model_loader.features),
        "features": model_loader.features
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
