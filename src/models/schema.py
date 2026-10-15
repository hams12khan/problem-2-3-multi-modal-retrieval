"""
Declarative base tables and workflow definitions using Pixeltable TableModel.
"""

from __future__ import annotations

import os
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions import openai

TableModel = pxt.model_base()


class FacilityVideos(TableModel, name="facility_videos"):
    """
    Base table storing untrimmed surveillance (UCF-Crime) and wearable bodycam (Ego4D)
    streams alongside B-tree indexed operational metadata.
    """
    video_id = pxt.Column(value=pxtf.uuid.uuid7())
    video: pxt.Video
    camera_zone: pxt.String        # e.g., 'dock_bay_4', 'chemical_storage', 'bodycam_shift_2'
    shift_date: pxt.String         # ISO date string (YYYY-MM-DD)
    anomaly_tag: pxt.String | None # Ground-truth UCF-Crime class or triage label

    __indexes__ = [
        pxt.BtreeIndex(camera_zone),
        pxt.BtreeIndex(shift_date),
    ]


class SafetyManuals(TableModel, name="safety_manuals"):
    """
    Base table storing OSHA 29 CFR 1910 regulatory manuals and plant SOP documents.
    """
    doc_id = pxt.Column(value=pxtf.uuid.uuid7())
    doc: pxt.Document
    regulation_code: pxt.String    # e.g., 'OSHA-1910.178-Forklifts', 'OSHA-1910.1200-HazCom'


class IncidentInvestigations(TableModel, name="investigations"):
    """
    Declarative workflow table: inserting a query row automatically executes hybrid
    multi-modal retrieval across visual, acoustic, and regulatory views and synthesizes
    the final forensic report.
    """
    investigation_id = pxt.Column(value=pxtf.uuid.uuid7())
    query_text: pxt.String
    target_zone: pxt.String
    evidence_frame: pxt.Image | None
