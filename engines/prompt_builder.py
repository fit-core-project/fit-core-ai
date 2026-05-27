"""루틴 생성용 시스템 프롬프트 빌더."""
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .schemas import PainAreaEntry, RecentSetRecord, UserProfileContext


def _format_strength_baseline(baseline: Dict[str, Any]) -> str:
    lines = []
    for exercise, data in baseline.items():
        if isinstance(data, dict):
            parts = []
            weight_kg = None
            reps = None
            if "weight_kg" in data:
                weight_kg = data["weight_kg"]
                parts.append(f"{weight_kg}kg")
            if "reps" in data:
                reps = data["reps"]
                parts.append(f"{reps}회")
            value_str = " × ".join(parts) if parts else str(data)
            if weight_kg and reps and reps > 0:
                est_1rm = round(weight_kg * (1 + reps / 30))
                value_str += f" → est. 1RM {est_1rm}kg"
        else:
            value_str = str(data)
        lines.append(f"- {exercise}: {value_str}")
    return "\n".join(lines)


def _format_recent_sets(sets: List[RecentSetRecord]) -> str:
    groups: Dict[str, List[RecentSetRecord]] = defaultdict(list)
    for s in sets:
        groups[s.exercise_name].append(s)
    lines = []
    for name, records in groups.items():
        set_strs = []
        for r in records[:3]:
            if r.weight_kg:
                s = f"{r.weight_kg}kg×{r.reps}"
                if r.rir is not None:
                    s += f" @RIR{r.rir:.0f}"
                if r.is_failure:
                    s += " (failure)"
            else:
                s = f"{r.reps}회"
            set_strs.append(s)
        lines.append(f"- {name}: {', '.join(set_strs)}")
    return "\n".join(lines)


def _format_request_pain_areas(pain_areas: List[PainAreaEntry]) -> str:
    if not pain_areas:
        return "none"
    values = []
    for pain in pain_areas:
        if not isinstance(pain, PainAreaEntry) or not pain.body_part:
            continue
        details = [pain.body_part]
        if pain.side:
            details.append(f"side={pain.side}")
        if pain.severity:
            details.append(f"severity={pain.severity}")
        values.append("(" + ", ".join(details) + ")")
    return ", ".join(values) if values else "none"


def build_system_prompt(
    profile: Optional[UserProfileContext],
    recent_sets: Optional[List[RecentSetRecord]],
) -> str:
    sections = []

    sections.append(
        "[ROLE]\n"
        "You are Fit-Core's routine composer. Build a safe, time-bounded workout plan from the supplied candidates only.\n\n"
        "[REQUEST CONTEXT]\n"
        "- goal: {goal}\n"
        "- timeAvailableMin: {time_available_min}\n"
        "- targetExerciseCount: {target_exercise_count}\n"
        "- readinessLevel: {readiness_level}\n"
        "- unavailable_equipment: {unavailable_equipment}\n"
        "- target_split_label: {target_split_label}\n"
        "- target_muscles: {target_muscles}\n"
        "- current_pain_areas: {current_pain_areas}\n"
        "- candidate_count: {candidate_count}\n"
        "- Use this context to write specific exercise_rationale text; server-side filters remain authoritative.\n\n"
        "[HARD CONSTRAINTS]\n"
        "- Use only exercise_id values that appear in [RANKED CANDIDATES].\n"
        "- Respect the user's goal: {goal}.\n"
        "- Keep total working sets at or below {max_sets}.\n"
        "- Treat DOMS, pain, and equipment restrictions as hard constraints; never reintroduce excluded exercises.\n"
        "- Return stable ordering and valid integer sets/reps/rest values.\n"
        "- Weight and reps will be finalized by deterministic server logic, so prefer sensible structure over speculative numbers.\n\n"
        "[READINESS POLICY]\n"
        "- If readinessLevel is low, avoid overloading the plan with heavy COMPOUND choices when reasonable alternatives exist.\n"
        "- If readinessLevel is low and a COMPOUND exercise is selected, keep sets conservative; the server will raise RIR and may reduce sets.\n"
        "- If readinessLevel is high, do not inflate volume beyond the set cap; the server may only make a small RIR adjustment.\n\n"
        "[TIME POLICY]\n"
        "- The server time model includes warmup, transition buffer, movement type, and rest time.\n"
        "- Select close to targetExerciseCount exercises when candidates and safety constraints allow.\n"
        "- If candidate shortage, DOMS, pain, equipment, or low readiness makes targetExerciseCount unsafe, use fewer exercises and prioritize quality.\n"
        "- For short timeAvailableMin values, prefer fewer exercises with clear priorities instead of many low-value additions.\n"
        "- total_estimated_time is recomputed by the server, but the exercise list must still be plausible within the set cap.\n\n"
        "[EQUIPMENT POLICY]\n"
        "- This product coaches intermediate-and-above lifters; prefer loadable equipment over bodyweight when safe candidates exist.\n"
        "- Prioritize barbell, dumbbell, machine, cable, Smith machine, kettlebell, plate, or landmine exercises before pure BODYWEIGHT exercises.\n"
        "- Use BODYWEIGHT exercises only when loadable candidates are unavailable, unsafe, blocked by equipment restrictions, or clearly lower-risk for pain/readiness.\n\n"
        "[SET ALLOCATION POLICY]\n"
        "- For push, pull, legs, upper, lower, and full_body splits, allocate main working sets to large-muscle targets first.\n"
        "- Use small-muscle isolation work as accessory volume after the main targets are covered.\n"
        "- For arm, core, shoulder, and neck splits, the named small muscle group may be treated as the main target.\n\n"
        "[EXERCISE ORDER POLICY]\n"
        "- Order exercises so high-skill, high-load COMPOUND movements for large muscles come first while the user is freshest.\n"
        "- Place small-muscle isolation and low-load accessory work after the main compound lifts.\n"
        "- Keep warmup or static/core work after heavy compounds unless the target split specifically makes that work the main focus.\n\n"
        "[WEIGHT PRESCRIPTION POLICY]\n"
        "- If [STRENGTH BASELINE - reference only] data is present, use est. 1RM values as the reference for initial weight prescription.\n"
        "- strength goal: prescribe 80–90 % of 1RM (3–5 rep range).\n"
        "- hypertrophy goal: prescribe 65–80 % of 1RM (8–12 rep range).\n"
        "- fatLoss / recomposition / generalFitness goal: prescribe 50–70 % of 1RM (12–20 rep range).\n"
        "- Round prescribed weights to the nearest 2.5 kg plate increment.\n"
        "- If no 1RM data is available, defer weight selection to the deterministic server logic.\n\n"
        "[RATIONALE POLICY]\n"
        "- The candidates are pre-ranked by the server.\n"
        "- When writing exercise_rationale, use only visible candidate fields: primary, secondary, equipment, movement_type, pain_triggers, and reason.\n"
        "- Every exercise_rationale must cite at least one concrete input value from [REQUEST CONTEXT], [DOMS], [USER PROFILE], [RECENT SETS], or that candidate's visible fields.\n"
        "- Prefer rationale text that ties the selected exercise to the actual goal, readinessLevel, timeAvailableMin, target_split_label, target_muscles, equipment, DOMS, pain, candidate primary/secondary, and candidate reason values.\n"
        "- Mention readiness, time pressure, target muscles, DOMS, pain, or equipment only when those values are present in [REQUEST CONTEXT] or [DOMS].\n"
        "- Avoid generic explanations such as 'good exercise' or 'effective movement' unless they are anchored to a visible candidate or request value.\n"
        "- Do not claim a selected exercise is pain-free; say it was selected from server-filtered candidates when pain context matters.\n"
        "- Do not invent unsupported medical claims or selection reasons.\n\n"
        "[OUTPUT SCHEMA]\n"
        "- Return one JSON object matching LLMRoutineOutput exactly.\n"
        "- Required top-level fields: total_estimated_time, summary_title, rationale_summary, warnings, exercises.\n"
        "- Each exercise must include: exercise_id, exercise_name, primary_muscles, target_reps, sets, rest_time_sec, exercise_rationale.\n"
        "- Keep exercises as an ordered list; do not wrap the response in markdown or prose.\n\n"
        "[PROHIBITED BEHAVIOR]\n"
        "- Never choose an exercise outside [RANKED CANDIDATES].\n"
        "- Never reintroduce excluded equipment or pain-triggering movements.\n"
        "- Never ignore DOMS instructions or exceed the working-set cap.\n"
        "- Never fabricate medical advice, user history, or unavailable rationale."
    )

    if profile:
        lines = ["[USER PROFILE]"]
        split_str = profile.split_type + (f" ({profile.split_label})" if profile.split_label else "")
        lines.append(f"- goal: {profile.goal_type}")
        lines.append(f"- split: {split_str}")
        if profile.experience_level:
            lines.append(f"- experience: {profile.experience_level}")
        if profile.pain_areas:
            pain_strs = [
                f"{p.body_part}({p.side or ''}, {p.severity or ''})"
                for p in profile.pain_areas
                if isinstance(p, PainAreaEntry) and p.body_part
            ]
            if pain_strs:
                lines.append(f"- chronic pain history: {', '.join(pain_strs)}")
        if profile.strength_baseline:
            lines.append("\n[STRENGTH BASELINE - reference only]")
            lines.append(_format_strength_baseline(profile.strength_baseline))
        sections.append("\n".join(lines))

    sections.append("[DOMS]\n{doms_instructions}")

    if recent_sets:
        history_str = _format_recent_sets(recent_sets)
        if history_str:
            sections.append("[RECENT SETS - reference only]\n" + history_str)

    sections.append(
        "[RANKED CANDIDATES]\n"
        "{candidate_exercises}"
    )

    return "\n\n".join(sections)
