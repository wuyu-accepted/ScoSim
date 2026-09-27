from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROMPT_FILES = {
    "ga_importance": "generative_agents_importance_v1.prompt.txt",
    "ga_reflection": "generative_agents_reflection_v1.prompt.txt",
    "be_argument_extraction": "belief_engine_argument_extraction_v1.prompt.txt",
    "be_evidence_judgment": "belief_engine_evidence_judgment_v1.prompt.txt",
    "chuang_confirmation_update": "chuang_confirmation_update_v1.prompt.txt",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenPrompt:
    prompt_id: str
    path: Path
    text: str
    sha256: str

    def render(self, replacements: Mapping[str, Any]) -> str:
        expected = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", self.text))
        supplied = set(replacements)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            raise ValueError(
                f"prompt replacement mismatch for {self.prompt_id}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        rendered = self.text
        for key, value in replacements.items():
            rendered = rendered.replace(
                "{{" + key + "}}",
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
        if re.search(r"\{\{[A-Z0-9_]+\}\}", rendered):
            raise ValueError(f"unresolved placeholder in prompt {self.prompt_id}")
        return rendered


class FrozenPromptRegistry:
    def __init__(self, prompt_root: Path, expected_hashes: Mapping[str, str]):
        self.prompts: dict[str, FrozenPrompt] = {}
        if set(expected_hashes) != set(PROMPT_FILES):
            raise ValueError("expected auxiliary prompt set drifted")
        for prompt_id, filename in PROMPT_FILES.items():
            path = prompt_root / filename
            text = path.read_text(encoding="utf-8")
            actual = sha256_text(text)
            if actual != expected_hashes[prompt_id]:
                raise ValueError(f"auxiliary prompt hash mismatch: {prompt_id}")
            self.prompts[prompt_id] = FrozenPrompt(prompt_id, path, text, actual)

    def __getitem__(self, prompt_id: str) -> FrozenPrompt:
        return self.prompts[prompt_id]
