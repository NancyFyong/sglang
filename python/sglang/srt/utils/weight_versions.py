from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Union

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

# ======================================================================
# Shared types
# ======================================================================
WeightVersionSpan = Dict[str, Union[str, int]]
WeightVersionSpans = List[WeightVersionSpan]


@dataclasses.dataclass(frozen=True, slots=True)
class WeightVersionEvent:
    old_version: str
    num_output_tokens: int


# ======================================================================
# Scheduler process
# ======================================================================
def record_weight_version_events(reqs: Iterable[Req], old_version: str) -> int:
    num_recorded = 0
    for req in reqs:
        if req.output_ids:
            req.weight_version_events.append(
                WeightVersionEvent(
                    old_version=old_version,
                    num_output_tokens=len(req.output_ids),
                )
            )
            num_recorded += 1
    return num_recorded


def compute_weight_version_spans(
    events: List[WeightVersionEvent],
    current_version: str,
    num_output_tokens: int,
) -> WeightVersionSpans:
    ends = [
        (event.old_version, min(event.num_output_tokens, num_output_tokens))
        for event in events
    ]
    ends.append((current_version, num_output_tokens))
    sampled_ends = [
        (version, end)
        for index, (version, end) in enumerate(ends)
        if index == 0 or end > ends[index - 1][1]
    ]
    boundaries = [
        (version, end)
        for index, (version, end) in enumerate(sampled_ends)
        if index == len(sampled_ends) - 1 or version != sampled_ends[index + 1][0]
    ]

    spans: WeightVersionSpans = []
    prev_end = 0
    for version, end in boundaries:
        spans.append({"version": version, "start": prev_end, "end": end})
        prev_end = end
    return spans


# ======================================================================
# TokenizerManager
# ======================================================================
def add_weight_versions_to_meta_info(
    meta_info: Dict[str, Any],
    spans: WeightVersionSpans,
    num_output_tokens: int,
) -> None:
    spans = [
        {**span, "end": min(span["end"], num_output_tokens)}
        for span in spans
        if span["start"] < num_output_tokens or span["start"] == 0
    ]

    meta_info["weight_versions"] = spans
    meta_info["weight_version"] = spans[-1]["version"]


# ======================================================================
# OpenAI-compatible endpoints
# ======================================================================
def build_endpoint_weight_version_metadata(meta_info: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {"weight_version": meta_info["weight_version"]}
    if "weight_versions" in meta_info:
        metadata["weight_versions"] = meta_info["weight_versions"]
    return metadata
