"""External learned initializer.

This is not a faithful L2WS reproduction. See states.L2WS_MAPPING_NOTE.
The class name and method label stay L2WS_INSPIRED.
"""

from __future__ import annotations

from ..states import L2WS_MAPPING_NOTE, L2WS_MAPPING_STATUS


class L2WSInspired:
    name = "C6_L2WS_INSPIRED"
    mapping_status = L2WS_MAPPING_STATUS

    def fit(self, rows: list[dict]) -> None:
        raise NotImplementedError(
            "Refusing to train an OSQP warm start under the L2WS name. "
            + L2WS_MAPPING_NOTE
        )

    def note(self) -> str:
        return L2WS_MAPPING_NOTE
