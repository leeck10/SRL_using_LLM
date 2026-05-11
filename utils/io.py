# -*- coding: utf-8 -*-
"""I/O utilities with automatic encoding detection.

Korean data files (e.g., framefiles) may be stored in cp949/euc-kr encoding
rather than UTF-8. These helpers try multiple encodings to load such files
transparently.
"""

import json
from typing import Any


def load_json(filepath: str) -> Any:
    """Load a JSON file with automatic encoding fallback.

    Tries UTF-8 first, then cp949 and euc-kr (common Korean encodings),
    and finally latin-1 as a last resort.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If all encoding attempts fail.
    """
    encodings = ["utf-8", "cp949", "euc-kr", "latin-1"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except json.JSONDecodeError:
            # If JSON parsing fails, it's not an encoding problem
            raise
    raise ValueError(
        f"Could not decode {filepath} with any of: {encodings}"
    )
