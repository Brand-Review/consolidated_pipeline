"""
Shared PromptRegistry — single source of truth for all LLM prompts.

Prompt files live in consolidated_pipeline/prompts/{name}_{version}.yaml.
Any module (brand_compliance_judge, vllm_analyzer, openrouter_analyzer, …)
resolves the directory via:
  1. BRANDGUARD_PROMPTS_DIR  environment variable   (explicit override)
  2. This file's location → ../../../../prompts/    (works when installed as a package)
  3. Caller-supplied prompts_dir argument

Versioning: bump version in the YAML filename (e.g. brand_compliance_judge_v2.yaml)
and pass version="v2" to get().  Active versions are controlled by environment
variables (PROMPT_VERSION_<NAME>, e.g. PROMPT_VERSION_BRAND_COMPLIANCE_JUDGE=v2)
or the default_versions dict below.

Langfuse integration: if a LangfuseClient is passed, prompts are fetched from
Langfuse Prompt Management first, with local YAML as fallback.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Default active version for each prompt name.
# Override per-prompt via env: PROMPT_VERSION_BRAND_COMPLIANCE_JUDGE=v2
# --------------------------------------------------------------------------- #
_DEFAULT_VERSIONS: Dict[str, str] = {
    "brand_compliance_judge": "v1",
    "copywriting_text": "v1",
    "copywriting_image_vllm": "v1",
    "copywriting_image_openrouter": "v1",
    "logo_judge": "v1",
    "logo_detector": "v1",
}


def _resolve_prompts_dir() -> Path:
    """Return the canonical prompts directory."""
    env_dir = os.environ.get("BRANDGUARD_PROMPTS_DIR")
    if env_dir:
        return Path(env_dir)
    # This file is at consolidated_pipeline/src/brandguard/core/prompt_registry.py
    # prompts/ is at consolidated_pipeline/prompts/
    return Path(__file__).parent.parent.parent.parent / "prompts"


@dataclass
class PromptTemplate:
    name: str
    version: str
    system: str
    user_template: str
    description: str = ""
    source: str = "local"          # "langfuse" | "local"
    # Extra fields from YAML (e.g. verdict_addendum for brand_compliance_judge)
    extras: Dict[str, str] = field(default_factory=dict)


class PromptRegistry:
    """
    Fetch and cache versioned prompt templates.

    Usage
    -----
    registry = PromptRegistry()
    tmpl = registry.get("brand_compliance_judge")
    system_msg = tmpl.system
    user_msg   = tmpl.user_template.format(brand_context=..., ...)
    addendum   = tmpl.extras.get("verdict_addendum", "")
    """

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        langfuse_client=None,
    ):
        self._dir = Path(prompts_dir) if prompts_dir else _resolve_prompts_dir()
        self._langfuse = langfuse_client
        self._cache: Dict[str, PromptTemplate] = {}
        logger.info(f"PromptRegistry: dir={self._dir}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        """
        Return a PromptTemplate by name (and optional version).

        Version resolution order:
          1. `version` argument
          2. PROMPT_VERSION_<NAME_UPPER> env var
          3. _DEFAULT_VERSIONS dict
          4. "v1"
        """
        if version is None:
            env_key = f"PROMPT_VERSION_{name.upper().replace('-', '_')}"
            version = os.environ.get(env_key) or _DEFAULT_VERSIONS.get(name, "v1")

        cache_key = f"{name}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._fetch_from_langfuse(name, version) or self._load_from_yaml(name, version)
        self._cache[cache_key] = prompt
        return prompt

    def active_version(self, name: str) -> str:
        """Return the currently configured version string for a prompt."""
        env_key = f"PROMPT_VERSION_{name.upper().replace('-', '_')}"
        return os.environ.get(env_key) or _DEFAULT_VERSIONS.get(name, "v1")

    def list_prompts(self) -> Dict[str, str]:
        """Return {name: active_version} for all known prompts."""
        return {name: self.active_version(name) for name in _DEFAULT_VERSIONS}

    def push_to_langfuse(self, name: str, version: Optional[str] = None) -> bool:
        """Upload a local YAML prompt to Langfuse Prompt Management."""
        if not self._langfuse or not getattr(self._langfuse, "enabled", False):
            logger.warning("Langfuse not configured — cannot push prompt.")
            return False
        version = version or self.active_version(name)
        try:
            tmpl = self._load_from_yaml(name, version)
        except FileNotFoundError as exc:
            logger.error(f"push_to_langfuse: {exc}")
            return False
        combined = f"SYSTEM:\n{tmpl.system}\n\nUSER TEMPLATE:\n{tmpl.user_template}"
        result = self._langfuse.create_prompt(name=name, prompt=combined, labels=["production"])
        if result is not None:
            logger.info(f"✅ Pushed prompt '{name}' ({version}) to Langfuse.")
            return True
        return False

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _fetch_from_langfuse(self, name: str, version: str) -> Optional[PromptTemplate]:
        if not self._langfuse or not getattr(self._langfuse, "enabled", False):
            return None
        try:
            raw = self._langfuse.get_prompt(name, version)
            if raw is None:
                return None
            compiled = raw.compile() if hasattr(raw, "compile") else str(raw)
            system, _, user_template = compiled.partition("\n\nUSER TEMPLATE:\n")
            system = system.replace("SYSTEM:\n", "", 1)
            return PromptTemplate(
                name=name, version=version,
                system=system.strip(), user_template=user_template.strip(),
                source="langfuse",
            )
        except Exception as exc:
            logger.debug(f"Langfuse fetch failed for '{name}' ({version}): {exc}")
            return None

    def _load_from_yaml(self, name: str, version: str) -> PromptTemplate:
        yaml_path = self._dir / f"{name}_{version}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Prompt YAML not found: {yaml_path}\n"
                f"Create the file or set BRANDGUARD_PROMPTS_DIR to the correct directory."
            )
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        # Collect any extra keys beyond the standard fields
        known = {"name", "version", "description", "system", "user_template"}
        extras = {k: v for k, v in data.items() if k not in known and isinstance(v, str)}

        return PromptTemplate(
            name=data.get("name", name),
            version=data.get("version", version),
            system=data.get("system", "").strip(),
            user_template=data.get("user_template", "").strip(),
            description=data.get("description", ""),
            source="local",
            extras=extras,
        )


# Module-level singleton — import and use directly:
#   from brandguard.core.prompt_registry import registry
#   tmpl = registry.get("brand_compliance_judge")
registry = PromptRegistry()
