"""
BrandComplianceJudge — Unified OpenRouter LLM brand compliance call.

Replaces:
  - PaddleOCR / Tesseract (text extraction)
  - HuggingFace font identifier (font identification)
  - HybridToneAnalyzer + per-analyzer copywriting call
  - Separate logo LLM judge in LogoDetector

One multimodal call that:
  1. Verifies YOLO logo detections (is each bbox a real logo?)
     → If YOLO produced no confident detections, LLM detects logos itself
  2. Verifies k-means dominant colors are accurate
  3. Extracts all text from the image (replaces PaddleOCR)
  4. Identifies fonts per text region (replaces font classifier model)
  5. Scores all 4 compliance dimensions against brand guidelines
  6. Returns verified/LLM-detected bboxes for YOLO fine-tuning
"""

import os
import base64
import json
import logging
from typing import Any, Dict, List, Optional

import requests

from .prompt_registry import registry as _prompt_registry

logger = logging.getLogger(__name__)

# Default model — override via OPENROUTER_MODEL env var
_DEFAULT_MODEL = "openai/gpt-5.1"

# Load prompts from the central registry at import time.
# To switch versions: set PROMPT_VERSION_BRAND_COMPLIANCE_JUDGE=v2
_PROMPT = _prompt_registry.get("brand_compliance_judge")
_SYSTEM_PROMPT = _PROMPT.system
_USER_PROMPT_TEMPLATE = _PROMPT.user_template
_VERDICT_TASK_ADDENDUM = _PROMPT.extras.get("verdict_addendum", "")


class BrandComplianceJudge:
    """
    Unified OpenRouter multimodal call for brand compliance scoring.

    Args:
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        model: OpenRouter model string. Falls back to OPENROUTER_MODEL env var,
               then to "openai/gpt-4o".
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)

    def run(
        self,
        image_path: str,
        brand_context: str,
        dominant_colors: List[Dict[str, Any]],
        logo_detections: List[Dict[str, Any]],
        brand_rules: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        verdict_mode: str = 'threshold',
    ) -> Optional[Dict[str, Any]]:
        """
        Run the unified brand compliance judge.

        Args:
            image_path: Absolute path to the image file.
            brand_context: RAG-retrieved brand guideline text (may be empty string
                           if no brand profile is available).
            brand_rules: Structured brand rules from brand_store (fonts, colors,
                         logo position, etc.). Injected as {brand_rules} in the
                         prompt so the LLM has authoritative, specific rules to
                         score against rather than inferring from examples.
            dominant_colors: k-means output list, e.g.
                             [{"hex": "#340081", "percentage": 42.3}, ...]
            logo_detections: YOLO raw detections, e.g.
                             [{"detection_id": 0, "bbox": [x1,y1,x2,y2],
                               "confidence": 0.87}, ...]
            few_shot_examples: Optional list from asset_rag.retrieve_similar().
                               Each entry: {"label": "approved"|"rejected",
                                            "image_b64": str,
                                            "rejection_reasons": [...]}

        Returns:
            Parsed JSON dict from the LLM, or None on failure.
        """
        if not self.api_key:
            logger.warning("[BrandComplianceJudge] No OPENROUTER_API_KEY — skipping LLM judge")
            return None

        try:
            image_b64 = self._encode_image(image_path)
            media_type = self._image_media_type(image_path)
        except Exception as e:
            logger.error(f"[BrandComplianceJudge] Failed to encode image: {e}")
            return None

        messages = self._build_messages(
            image_b64=image_b64,
            media_type=media_type,
            brand_context=brand_context or "(No brand guidelines available — score based on general brand design standards)",
            brand_rules=brand_rules or {},
            dominant_colors=dominant_colors,
            logo_detections=logo_detections,
            few_shot_examples=few_shot_examples or [],
            verdict_mode=verdict_mode,
        )

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
                timeout=120,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            logger.info(
                "[BrandComplianceJudge] Scores — color=%.2f logo=%.2f typography=%.2f copywriting=%.2f",
                result.get("color", {}).get("score", 0),
                result.get("logo", {}).get("score", 0),
                result.get("typography", {}).get("score", 0),
                result.get("copywriting", {}).get("score", 0),
            )
            return result
        except Exception as e:
            logger.error(f"[BrandComplianceJudge] LLM call failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Base64-encode an image for the multimodal payload."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _image_media_type(image_path: str) -> str:
        ext = image_path.rsplit(".", 1)[-1].lower()
        return {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "gif": "image/gif",
        }.get(ext, "image/jpeg")

    @staticmethod
    def _format_brand_rules(brand_rules: Dict[str, Any]) -> str:
        """
        Format the structured brand_rules dict (from brand_store) into a
        human-readable bullet list for the LLM prompt.

        Each rule is only emitted when the stored value is non-empty/non-None,
        so missing fields don't produce misleading bullets.
        """
        lines: List[str] = []

        typo = brand_rules.get("typography_rules", {})
        if typo.get("bangla_font"):
            lines.append(f"- Approved Bangla font: {typo['bangla_font']} only")
        if typo.get("english_font"):
            lines.append(f"- Approved English font: {typo['english_font']} only")
        if typo.get("approved_fonts"):
            lines.append(f"- Other approved fonts: {', '.join(typo['approved_fonts'])}")

        color = brand_rules.get("color_rules", {})
        if color.get("palette"):
            lines.append(f"- Brand color palette: {', '.join(color['palette'])}")
        if color.get("gradient"):
            lines.append(f"- Required gradient: {color['gradient']}")
        if color.get("forbidden"):
            lines.append(f"- Forbidden colors: {', '.join(color['forbidden'])}")

        logo = brand_rules.get("logo_rules", {})
        if logo.get("position"):
            lines.append(f"- Logo position: {logo['position']} corner always")
        if logo.get("min_height_px"):
            lines.append(f"- Logo minimum height: {logo['min_height_px']}px")
        if logo.get("dark_bg_use_white"):
            lines.append("- Logo version rule: use white logo on dark/colored backgrounds")
        if logo.get("colorful_on_white_only"):
            lines.append("- Logo version rule: use colorful logo on white background only")

        voice = brand_rules.get("brand_voice_rules", {})
        if voice.get("tone"):
            lines.append(f"- Brand tone: {voice['tone']}")
        if voice.get("language"):
            lines.append(f"- Primary language: {voice['language']}")

        return "\n".join(lines) if lines else "(No structured rules stored — rely on guidelines below)"

    def _build_messages(
        self,
        image_b64: str,
        media_type: str,
        brand_context: str,
        brand_rules: Dict[str, Any],
        dominant_colors: List[Dict],
        logo_detections: List[Dict],
        few_shot_examples: List[Dict],
        verdict_mode: str = 'threshold',
    ) -> List[Dict]:
        messages: List[Dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Few-shot turns (approved first, then rejected)
        approved = [e for e in few_shot_examples if e.get("label") == "approved"]
        rejected = [e for e in few_shot_examples if e.get("label") == "rejected"]

        for ex in approved[:2]:
            if ex.get("image_b64"):
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Example: APPROVED brand asset"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{ex['image_b64']}"},
                        },
                    ],
                })
                messages.append({
                    "role": "assistant",
                    "content": '{"note": "This asset is approved — it follows brand guidelines."}',
                })

        for ex in rejected[:2]:
            if ex.get("image_b64"):
                reasons = ex.get("rejection_reasons", [])
                reasons_text = "; ".join(reasons) if reasons else "Does not meet brand standards"
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Example: REJECTED brand asset. Reasons: {reasons_text}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{ex['image_b64']}"},
                        },
                    ],
                })
                messages.append({
                    "role": "assistant",
                    "content": f'{{"note": "This asset is rejected. Reasons: {reasons_text}."}}',
                })

        # Main analysis turn
        base_prompt = _USER_PROMPT_TEMPLATE.format(
            brand_rules=self._format_brand_rules(brand_rules),
            brand_context=brand_context,
            logo_detections_json=json.dumps(logo_detections, indent=2) if logo_detections else "[]",
            dominant_colors_json=json.dumps(dominant_colors, indent=2) if dominant_colors else "[]",
        )
        # In llm mode, ask the LLM to also provide an explicit verdict + verdict_reason
        if verdict_mode == 'llm':
            base_prompt = base_prompt + _VERDICT_TASK_ADDENDUM

        user_content = [
            {
                "type": "text",
                "text": base_prompt,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
            },
        ]
        messages.append({"role": "user", "content": user_content})
        return messages
