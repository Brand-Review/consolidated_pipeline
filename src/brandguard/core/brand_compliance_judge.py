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

from .llm_client import LLMClient, LLMResponseError
from .prompt_registry import registry as _prompt_registry

logger = logging.getLogger(__name__)

# Default model — override via OPENROUTER_MODEL env var
_DEFAULT_MODEL = "openai/gpt-5.1"

# Load prompts from the central registry at import time.
# To switch versions: set PROMPT_VERSION_BRAND_COMPLIANCE_JUDGE=v2
_PROMPT = _prompt_registry.get("brand_compliance_judge")
_SYSTEM_PROMPT = _PROMPT.system
_USER_PROMPT_TEMPLATE_STATIC = _PROMPT.extras.get("user_template_static", "")
_USER_PROMPT_TEMPLATE_DYNAMIC = _PROMPT.extras.get("user_template_dynamic", "")
_VERDICT_TASK_ADDENDUM = _PROMPT.extras.get("verdict_addendum", "")


def _is_anthropic_model(model: str) -> bool:
    """Return True when the model routes through Anthropic, enabling cache_control blocks."""
    m = model.lower()
    return "anthropic/" in m or m.startswith("claude")


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
        # Populated after each run() call; readable by eval harnesses for token/cost tracking
        self.last_usage: Dict[str, Any] = {}
        self.llm = LLMClient(api_key=self.api_key, model=self.model)

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
            result, usage = self.llm.chat(
                messages, response_format={"type": "json_object"}
            )
            self.last_usage = usage
            logger.info(
                "[BrandComplianceJudge] Scores — color=%.2f logo=%.2f typography=%.2f copywriting=%.2f",
                result.get("color", {}).get("score", 0),
                result.get("logo", {}).get("score", 0),
                result.get("typography", {}).get("score", 0),
                result.get("copywriting", {}).get("score", 0),
            )
            return result
        except LLMResponseError as e:
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
        use_cache = _is_anthropic_model(self.model)

        # System message — wrap in a content block for Anthropic cache_control
        if use_cache:
            system_msg: Dict[str, Any] = {
                "role": "system",
                "content": [{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral" , "ttl": "1h"}}],
            }
        else:
            system_msg = {"role": "system", "content": _SYSTEM_PROMPT}
        messages: List[Dict] = [system_msg]

        # Few-shot turns (approved first, then rejected).
        # Track user-message indices so we can mark the last one for caching.
        approved = [e for e in few_shot_examples if e.get("label") == "approved"]
        rejected = [e for e in few_shot_examples if e.get("label") == "rejected"]
        few_shot_user_indices: List[int] = []

        for ex in approved[:2]:
            if ex.get("image_b64"):
                few_shot_user_indices.append(len(messages))
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
                few_shot_user_indices.append(len(messages))
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

        # Cache breakpoint 2: mark last content block of last few-shot user message
        if use_cache and few_shot_user_indices:
            last_idx = few_shot_user_indices[-1]
            messages[last_idx]["content"][-1]["cache_control"] = {"type": "ephemeral","ttl": "1h"}

        # Strip internal-only fields from dominant_colors before serialization —
        # rgb is redundant with hex; cluster_id is an internal bookkeeping index.
        stripped_colors = [
            {k: c[k] for k in ("hex", "percentage", "saliency_weight") if k in c}
            for c in dominant_colors
        ]

        static_text = _USER_PROMPT_TEMPLATE_STATIC.format(
            brand_rules=self._format_brand_rules(brand_rules),
            brand_context=brand_context,
        )
        dynamic_text = _USER_PROMPT_TEMPLATE_DYNAMIC.format(
            logo_detections_json=json.dumps(logo_detections, indent=2) if logo_detections else "[]",
            dominant_colors_json=json.dumps(stripped_colors, indent=2) if stripped_colors else "[]",
        )
        if verdict_mode == 'llm':
            dynamic_text = dynamic_text + _VERDICT_TASK_ADDENDUM

        if use_cache:
            # Cache breakpoint 3: brand rules + context are static per brand → cacheable
            user_content: List[Dict] = [
                {"type": "text", "text": static_text, "cache_control": {"type": "ephemeral" , "ttl": "1h"}},
                {"type": "text", "text": dynamic_text},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
            ]
        else:
            user_content = [
                {"type": "text", "text": static_text + "\n\n" + dynamic_text},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
            ]

        messages.append({"role": "user", "content": user_content})
        return messages
