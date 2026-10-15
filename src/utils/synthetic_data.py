"""
Synthetic asset generation and curated benchmark scenarios for testing and demonstration.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw

def generate_synthetic_safety_assets(work_dir: Path) -> dict[str, Path]:
    """
    Creates a realistic self-contained OSHA regulatory document and synthetic video/audio
    stream if the user has not yet mounted the full 128-hour UCF-Crime / Ego4D archive locally.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. OSHA 29 CFR 1910 Regulatory Standard Text Document
    osha_doc_path = work_dir / "osha_1910_industrial_standards.txt"
    osha_content = (
        "# OSHA 29 CFR 1910.178 - Powered Industrial Trucks & Forklift Operations\n"
        "Section 1910.178(p)(1): If at any time a powered industrial truck is found to be in need of repair, "
        "defective, or in any way unsafe, the truck shall be taken out of service until it has been restored "
        "to safe operating condition. Hydraulic line leaks or brake skidding on loading docks require immediate "
        "Lockout/Tagout (LOTO) and floor spill containment.\n\n"
        "# OSHA 29 CFR 1910.1200 - Hazard Communication & Chemical Drum Storage\n"
        "Section 1910.1200(f)(6): Workplace chemical drums and hazardous materials containers in chemical_storage "
        "must be clearly labeled and protected against impact by forklifts or material handling equipment. "
        "Spilled corrosive or flammable liquids require immediate evacuation of the 15-foot radius and hazmat triage.\n\n"
        "# OSHA 29 CFR 1910.38 - Emergency Action Plans & Thermal/Explosion Hazards\n"
        "Section 1910.38(c): Facilities must maintain audible two-way radio alerting and visual strobe alarms "
        "for incipient arson, electrical arc flash, or pressure vessel explosions.\n\n"
        "# OSHA 29 CFR 1910.147 - The Control of Hazardous Energy (Lockout/Tagout)\n"
        "Section 1910.147(c)(4)(i): Procedures shall be developed, documented and utilized for the control of "
        "potentially hazardous energy when employees are engaged in activities covered by this section."
    )
    osha_doc_path.write_text(osha_content, encoding="utf-8")

    # 2. 16kHz WAV Audio Stream with an intentional silence gap for audio_splitter
    wav_path = work_dir / "bodycam_radio_callout.wav"
    sample_rate = 16000
    duration_sec = 6
    with wave.open(str(wav_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for i in range(sample_rate * duration_sec):
            t = i / sample_rate
            # 0.6s quiet stretch between 2.5s and 3.1s to validate min_silence_len=0.3
            if 2.5 <= t <= 3.1:
                sample = 0
            else:
                sample = int(8000.0 * math.sin(2.0 * math.pi * 440.0 * t))
            frames.extend(struct.pack("<h", sample))
        wav_file.writeframes(bytes(frames))

    # 3. Synthetic CCTV Frame (640x360 PNG) with simulated hazard markings
    img_path = work_dir / "dock_bay_4_hazard_frame.png"
    img = Image.new("RGB", (640, 360), color=(20, 24, 34))
    draw = ImageDraw.Draw(img)
    # Draw floor lane
    draw.polygon([(0, 360), (240, 160), (400, 160), (640, 360)], fill=(32, 38, 52))
    # Draw yellow hazard stripes
    draw.line([(80, 340), (260, 170)], fill=(245, 158, 11), width=4)
    draw.line([(560, 340), (380, 170)], fill=(245, 158, 11), width=4)
    # Draw forklift bounding box area
    draw.rectangle([(200, 180), (380, 300)], outline=(16, 185, 129), width=2)
    # Draw spill puddle area
    draw.ellipse([(260, 280), (420, 330)], fill=(239, 68, 68, 128), outline=(244, 63, 94))
    img.save(str(img_path))

    return {
        "osha_doc": osha_doc_path,
        "sample_wav": wav_path,
        "sample_img": img_path,
    }


def get_curated_scenarios() -> list[dict[str, Any]]:
    """
    Curated benchmark scenarios modeled on UCF-Crime surveillance and Ego4D bodycams.
    """
    return [
        {
            "id": "forklift_leak",
            "title": "Forklift Hydraulic Skid & HazMat Spill",
            "tag": "UCF-Crime: RoadAccidents042",
            "zone": "dock_bay_4",
            "shift_date": "2026-10-15",
            "uri": "s3://industrial-safety-vault/production/uploads/RoadAccidents042_x264.mp4",
            "query": "forklift skidding near chemical storage drums while supervisor reports hydraulic fluid leak over radio",
            "audio_window": "14.2s – 29.8s",
            "audio_score": 0.684,
            "transcript": "Supervisor on Channel 2: Watch out on Dock Bay 4, hydraulic line burst under the reach truck, fluid on the concrete apron near the blue drums!",
            "osha_code": "OSHA 29 CFR 1910.178(p)(1)",
            "osha_passage": "If at any time a powered industrial truck is found to be in need of repair, defective, or in any way unsafe, the truck shall be taken out of service until restored to safe operating condition.",
            "keyframes": [
                {"pos": 18, "pts": 18.0, "sim": 0.342, "objs": ["forklift: 0.96", "chemical_drum: 0.91", "oil_spill: 0.84"]},
                {"pos": 19, "pts": 19.0, "sim": 0.319, "objs": ["forklift: 0.95", "operator: 0.89", "chemical_drum: 0.88"]},
                {"pos": 15, "pts": 15.0, "sim": 0.274, "objs": ["forklift: 0.92", "pallet_rack: 0.86"]},
            ],
            "report": {
                "window": "14.2s to 29.8s (Peak Visual Keyframe @ PTS = 18.0s, Frame #18)",
                "root_cause": "Visual keyframe evidence (CLIP sim = 0.342) and DETR bounding boxes confirm a reach forklift skidding within 1.4 meters of staged chemical drums in dock_bay_4. Synchronized silence-snapped Whisper ASR (14.2s–29.8s, MiniLM sim = 0.684) corroborates a ruptured hydraulic line spraying fluid onto the concrete apron prior to loss of traction.",
                "violation": "OSHA 29 CFR 1910.178(p)(1) (Powered Industrial Trucks — Defective Equipment) & 1910.22(a)(2) (Walking-Working Surfaces kept free of slip hazards). Estimated direct/indirect incident risk mitigated: $42,000.",
                "action": "1) Immediately execute Lockout/Tagout (LOTO) on Forklift #402; 2) Deploy HazMat absorbent booms around Bay 4 apron; 3) Inspect hydraulic hose crimps across fleet before Shift 3."
            }
        },
        {
            "id": "chemical_arson",
            "title": "Thermal Ignition & Chemical Vapour Flash",
            "tag": "UCF-Crime: Explosion017",
            "zone": "chemical_storage",
            "shift_date": "2026-10-14",
            "uri": "s3://industrial-safety-vault/production/uploads/Explosion017_x264.mp4",
            "query": "sparks and smoke plume rising near solvent tank manifold with emergency evacuation callout",
            "audio_window": "42.0s – 68.5s",
            "audio_score": 0.719,
            "transcript": "Control Room Alert: Thermal sensor spike in Chemical Storage Bay B! Visible smoke near the MEK solvent transfer manifold—initiate sector evacuation!",
            "osha_code": "OSHA 29 CFR 1910.106(e)(6)(i)",
            "osha_passage": "Adequate precautions shall be taken to prevent the ignition of flammable vapors. Sources of ignition include open flames, lightning, smoking, cutting and welding, hot surfaces, frictional heat, and static, electrical, and mechanical sparks.",
            "keyframes": [
                {"pos": 51, "pts": 51.0, "sim": 0.368, "objs": ["smoke_plume: 0.94", "chemical_tank: 0.93", "arc_flash: 0.87"]},
                {"pos": 48, "pts": 48.0, "sim": 0.311, "objs": ["chemical_tank: 0.92", "operator_running: 0.85"]},
                {"pos": 55, "pts": 55.0, "sim": 0.289, "objs": ["smoke_plume: 0.97", "strobe_beacon: 0.81"]},
            ],
            "report": {
                "window": "42.0s to 68.5s (Peak Visual Keyframe @ PTS = 51.0s, Frame #51)",
                "root_cause": "CLIP semantic retrieval (sim = 0.368) localized an Electrostatic Discharge (ESD) arc flash and dense smoke plume at the MEK solvent transfer manifold in chemical_storage. Whisper ASR captured the synchronized Control Room radio warning at 42.0s.",
                "violation": "OSHA 29 CFR 1910.106(e)(6)(i) (Flammable Liquids — Grounding & Ignition Control) & 1910.38(c) (Emergency Action Plan).",
                "action": "1) Trigger automatic nitrogen purge and deluge valve in Bay B; 2) Verify bonding/grounding straps on all Class IB solvent drums; 3) Export keyframes via .to_coco_dataset() to retrain early smoke detector."
            }
        },
        {
            "id": "ego4d_bodycam",
            "title": "Egocentric Bodycam: Unclipped Harness & Guard Bypass",
            "tag": "Ego4D: EpisodicMemory_Shift2",
            "zone": "bodycam_shift_2",
            "shift_date": "2026-10-15",
            "uri": "s3://industrial-safety-vault/production/uploads/Ego4D_Maintenance_089.mp4",
            "query": "first-person view of technician reaching past interlocked safety light curtain without lockout padlock",
            "audio_window": "112.4s – 140.1s",
            "audio_score": 0.641,
            "transcript": "Technician Bodycam Mic: I am just going to clear this jammed conveyor sensor real quick without shutting down the main breaker—hand me the long pliers.",
            "osha_code": "OSHA 29 CFR 1910.147(c)(4)(i)",
            "osha_passage": "Procedures shall be developed, documented and utilized for the control of potentially hazardous energy when employees are engaged in the activities covered by this section (Lockout/Tagout).",
            "keyframes": [
                {"pos": 124, "pts": 124.0, "sim": 0.355, "objs": ["gloved_hand: 0.97", "light_curtain: 0.91", "conveyor_gear: 0.88"]},
                {"pos": 121, "pts": 121.0, "sim": 0.304, "objs": ["open_interlock_gate: 0.93", "pliers: 0.86"]},
                {"pos": 127, "pts": 127.0, "sim": 0.279, "objs": ["gloved_hand: 0.95", "energized_panel: 0.89"]},
            ],
            "report": {
                "window": "112.4s to 140.1s (Peak Visual Keyframe @ PTS = 124.0s, Frame #124)",
                "root_cause": "Egocentric wearable stream (bodycam_shift_2) shows operator hand entering an energized conveyor pinch-point past a bypassed optical light curtain (CLIP sim = 0.355). Silence-snapped bodycam audio confirms intentional bypass of Lockout/Tagout (LOTO) procedure.",
                "violation": "OSHA 29 CFR 1910.147(c)(4)(i) (Control of Hazardous Energy — LOTO) & 1910.212(a)(1) (Machine Guarding).",
                "action": "1) Halt Line 2 for tamper-proof interlock switch upgrade; 2) Conduct mandatory LOTO refresher with Shift 2 maintenance crew; 3) Archive near-miss clip for safety briefing."
            }
        }
    ]
