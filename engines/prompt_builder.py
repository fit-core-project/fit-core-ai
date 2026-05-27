"""Routine generation system prompt builder."""
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
                parts.append(f"{reps} reps")
            value_str = " x ".join(parts) if parts else str(data)
            if weight_kg and reps and reps > 0:
                est_1rm = round(weight_kg * (1 + reps / 30))
                value_str += f" -> est. 1RM {est_1rm}kg"
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
                s = f"{r.weight_kg}kg x {r.reps}"
                if r.rir is not None:
                    s += f" @RIR{r.rir:.0f}"
                if r.is_failure:
                    s += " (failure)"
            else:
                s = f"{r.reps} reps"
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
        "- target_reps, sets, and rest_time_sec are required schema fields; provide plausible intent values consistent with the goal and movement type.\n"
        "- The server overrides target_reps and rest_time_sec with deterministic tables (goal x movement_type); your values serve as structural hints only.\n"
        "- target_weight_kg: set to null by default. The server resolves weight from recent sets and strength baseline; your value is used only when no historical data exists.\n"
        "- Do not perform 1RM percentage calculations; all weight math is handled server-side.\n"
        "- Always set target_weight_kg to null for BODYWEIGHT equipment.\n\n"
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
        "[EXAMPLES]\n"
        "Note: EXAMPLE_A/B/C IDs are illustration only; use only IDs from [RANKED CANDIDATES].\n\n"
        "Example 1 - hypertrophy / push / barbell available / no constraints:\n"
        '{{"total_estimated_time":45,"summary_title":"Push Hypertrophy","rationale_summary":["Barbell bench press leads compound chest work.","Lateral raise adds shoulder isolation."],"warnings":[],"exercises":[{{"exercise_id":"EXAMPLE_A","exercise_name":"Barbell Bench Press","primary_muscles":["chest"],"target_reps":10,"sets":3,"rest_time_sec":90,"target_weight_kg":null,"exercise_rationale":"COMPOUND chest primary; barbell available; hypertrophy; server finalizes weight."}},{{"exercise_id":"EXAMPLE_B","exercise_name":"Dumbbell Lateral Raise","primary_muscles":["side-deltoids"],"target_reps":12,"sets":2,"rest_time_sec":60,"target_weight_kg":null,"exercise_rationale":"ISOLATION shoulder accessory; placed after compound."}}]}}\n\n'
        "Example 2 - barbell blocked + shoulder pain -> BODYWEIGHT substitute, reduced volume:\n"
        '{{"total_estimated_time":30,"summary_title":"Push Constraint Safe","rationale_summary":["Barbell blocked; server-filtered candidates exclude barbell exercises.","Shoulder pain excluded overhead movements; exercise count reduced."],"warnings":["Barbell unavailable; bodyweight substitute used."],"exercises":[{{"exercise_id":"EXAMPLE_C","exercise_name":"Push-up","primary_muscles":["chest"],"target_reps":10,"sets":3,"rest_time_sec":75,"target_weight_kg":null,"exercise_rationale":"BODYWEIGHT from server-filtered candidates; barbell blocked and shoulder pain excluded loadable alternatives; target_weight_kg null."}}]}}\n\n'
        "[FINAL SELECTION REMINDER]\n"
        "In the actual response, never use EXAMPLE_* ids. Use only exercise_id values from [RANKED CANDIDATES].\n\n"
        "[PROHIBITED BEHAVIOR]\n"
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
