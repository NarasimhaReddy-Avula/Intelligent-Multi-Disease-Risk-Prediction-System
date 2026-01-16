"""
FastAPI Application Entry Point
Multi-Disease AI Healthcare Platform API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Multi-Disease AI Healthcare API",
    description="Predicts risks for Heart Disease, Diabetes, Breast Cancer, and Liver Disease",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "Multi-Disease AI Healthcare API"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Multi-Disease AI Healthcare API",
        "docs": "/docs",
        "health": "/health"
    }


# TODO: Import and include routers
# from .routes import predict, explain_shap, explain_nlp
# app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
# app.include_router(explain_shap.router, prefix="/explain", tags=["Explainability"])
# app.include_router(explain_nlp.router, prefix="/explain-nlp", tags=["LLM Explanations"])
