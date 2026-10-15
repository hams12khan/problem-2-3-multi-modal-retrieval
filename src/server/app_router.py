"""
FastAPI HTTP Serving and REST Endpoints via Pixeltable FastAPIRouter.
"""

from __future__ import annotations

import os
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import pixeltable as pxt
from pixeltable.serving import FastAPIRouter

from src.config import DEFAULT_NAMESPACE, API_HOST, API_PORT, UI_DIR
from src.models.schema import FacilityVideos, IncidentInvestigations
from src.retrieval.engine import execute_hybrid_retrieval
from src.synthesis.forensic_reporter import generate_forensic_report

# Initialize Pixeltable FastAPIRouter
api = FastAPIRouter(name="safety_rag_api")

# Declarative schema routes if tables are active
try:
    api.add_insert_route(
        FacilityVideos,
        path="/videos",
        inputs=[
            FacilityVideos.video,
            FacilityVideos.camera_zone,
            FacilityVideos.shift_date,
            FacilityVideos.anomaly_tag,
        ],
        outputs=[FacilityVideos.video_id],
    )
    api.add_compute_route(
        IncidentInvestigations,
        path="/investigations/compute",
        inputs=[
            IncidentInvestigations.query_text,
            IncidentInvestigations.target_zone,
            IncidentInvestigations.evidence_frame,
        ],
        outputs=[
            IncidentInvestigations.investigation_id,
            IncidentInvestigations.forensic_report,
        ],
    )
except Exception:
    pass


class InvestigationRequest(BaseModel):
    query_text: str
    target_zone: str = "dock_bay_4"
    evidence_frame: str | None = None


def create_app() -> FastAPI:
    """
    Constructs the production FastAPI application, mounting Pixeltable routes,
    custom synchronous endpoints, and the interactive web workbench.
    """
    app = FastAPI(
        title="SentinelPXT — Industrial Safety Video RAG API",
        description="Declarative Multimodal Video Retrieval & OSHA Forensic Triage with Pixeltable",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["System"])
    def health_check() -> dict[str, str]:
        return {"status": "healthy", "engine": "Pixeltable 0.7+", "service": "SentinelPXT"}

    @app.post("/investigate", tags=["Forensics"])
    def investigate_incident(req: InvestigationRequest) -> dict[str, Any]:
        """
        Synchronous def endpoint handler to ensure FastAPI dispatches requests to an
        isolated worker thread pool, preventing event-loop blocking during Pixeltable operations.
        """
        try:
            retrieval_res = execute_hybrid_retrieval(
                query_text=req.query_text,
                target_zone=req.target_zone,
            )
            report = generate_forensic_report(
                incident_query=req.query_text,
                target_zone=req.target_zone,
                visual_hits=retrieval_res["visual_hits"],
                audio_hits=retrieval_res["audio_hits"],
                osha_hits=retrieval_res["osha_hits"],
            )
            return {
                "query": req.query_text,
                "target_zone": req.target_zone,
                "elapsed_ms": retrieval_res["elapsed_ms"],
                "visual_context": retrieval_res["visual_hits"],
                "audio_context": retrieval_res["audio_hits"],
                "osha_context": retrieval_res["osha_hits"],
                "forensic_report": report,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/ui", response_class=HTMLResponse, tags=["UI"])
    def serve_workbench() -> HTMLResponse:
        index_file = UI_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>SentinelPXT Workbench UI not found</h1>", status_code=404)

    return app


def run_server(host: str = API_HOST, port: int = API_PORT) -> None:
    """Runs the FastAPI server via Uvicorn."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)

