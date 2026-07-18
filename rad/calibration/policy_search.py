from __future__ import annotations

from typing import Any

from rad.models.policy import PolicyProfile


def _is_dominated(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True if a is dominated by b on (ap_drop, false_safe, -expected_depth).

    Prefer lower AP drop, lower false-safe rate, and lower expected depth.
    """
    return (
        b["pixel_ap_drop"] <= a["pixel_ap_drop"]
        and b["false_safe_exit_rate"] <= a["false_safe_exit_rate"]
        and b["expected_depth"] <= a["expected_depth"]
        and (
            b["pixel_ap_drop"] < a["pixel_ap_drop"]
            or b["false_safe_exit_rate"] < a["false_safe_exit_rate"]
            or b["expected_depth"] < a["expected_depth"]
        )
    )


def feasible_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_pixel_ap_drop: float,
    max_false_safe_exit_rate: float,
) -> list[dict[str, Any]]:
    return [
        c
        for c in candidates
        if float(c["pixel_ap_drop"]) <= max_pixel_ap_drop
        and float(c["false_safe_exit_rate"]) <= max_false_safe_exit_rate
    ]


def pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for c in candidates:
        if any(_is_dominated(c, o) for o in candidates):
            continue
        front.append(c)
    return front


def _to_profile(name: str, row: dict[str, Any]) -> PolicyProfile:
    kwargs = {
        "gain_threshold": float(row["gain_threshold"]),
        "kappa": float(row["kappa"]),
        "map_uncertainty_threshold": float(row["map_uncertainty_threshold"]),
        "image_confidence_margin": float(row["image_confidence_margin"]),
        "stability_threshold": float(row["stability_threshold"]),
    }
    if name == "aggressive":
        return PolicyProfile.aggressive(
            gain_threshold=kwargs["gain_threshold"], kappa=kwargs["kappa"]
        )
    if name == "balanced":
        return PolicyProfile.balanced(
            gain_threshold=kwargs["gain_threshold"],
            kappa=kwargs["kappa"],
            map_uncertainty_threshold=kwargs["map_uncertainty_threshold"],
            image_confidence_margin=kwargs["image_confidence_margin"],
        )
    if name == "conservative":
        return PolicyProfile.conservative(**kwargs)
    raise ValueError(f"unknown profile name: {name}")


def select_three_profiles(pareto: list[dict[str, Any]]) -> dict[str, PolicyProfile]:
    """Pick conservative / balanced / aggressive operating points from Pareto set."""
    if not pareto:
        raise ValueError("pareto set is empty")
    # Aggressive: lowest expected depth among feasible
    aggressive_row = min(pareto, key=lambda r: (r["expected_depth"], r["false_safe_exit_rate"]))
    # Conservative: lowest false-safe then AP drop
    conservative_row = min(
        pareto, key=lambda r: (r["false_safe_exit_rate"], r["pixel_ap_drop"], -r["expected_depth"])
    )
    # Balanced: closest to median expected depth
    depths = sorted(float(r["expected_depth"]) for r in pareto)
    median = depths[len(depths) // 2]
    balanced_row = min(
        pareto,
        key=lambda r: (
            abs(float(r["expected_depth"]) - median),
            r["false_safe_exit_rate"],
            r["pixel_ap_drop"],
        ),
    )
    return {
        "aggressive": _to_profile("aggressive", aggressive_row),
        "balanced": _to_profile("balanced", balanced_row),
        "conservative": _to_profile("conservative", conservative_row),
    }


def search_policy_profiles(
    candidates: list[dict[str, Any]],
    *,
    max_pixel_ap_drop: float,
    max_false_safe_exit_rate: float,
) -> dict[str, Any]:
    feasible = feasible_candidates(
        candidates,
        max_pixel_ap_drop=max_pixel_ap_drop,
        max_false_safe_exit_rate=max_false_safe_exit_rate,
    )
    if not feasible:
        raise ValueError("no feasible candidates under constraints")
    pareto = pareto_front(feasible)
    profiles = select_three_profiles(pareto)
    return {"feasible": feasible, "pareto": pareto, "profiles": profiles}
