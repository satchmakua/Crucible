"""Pull the code solution out of a model response.

Prefers the last fenced code block (```python … ``` or a bare ``` … ```); if there's no
fence, treats the whole response as code (small models often skip the fence).
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Return the candidate code from `text` (last fenced block, else the whole text)."""
    blocks: list[str] = _FENCE.findall(text)
    if blocks:
        return str(blocks[-1]).strip()
    return text.strip()
