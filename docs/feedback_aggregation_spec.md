# P2-1 Feedback Aggregation Spec

## 1. Design Summary

P2 feedback-aware ranking must not feed raw `routine_feedback` rows directly into candidate ranking. The first implementation should convert append-only feedback rows into deterministic exercise-level aggregate signals, then expose bounded score adjustments behind a disabled-by-default feature flag.

V1 recommendation:

- Add `engines/feedback_aggregation.py`.
- Keep `ENABLE_FEEDBACK_AWARE_RANKING=false` by default.
- Use on-the-fly aggregation over `routine_feedback` for P2-2.
- Compute separate user-level and global-level exercise adjustments.
- Blend adjustments as `user_adjustment + 0.3 * global_adjustment`, clamped.
- Apply adjustments only after existing hard filters have excluded unsafe candidates.
- Never use `user_note` or any free text in aggregation, ranking, telemetry, or logs.

P2-1 is spec-only. P2-2 should implement aggregation helpers and candidate ranker integration.

## 2. Current Feedback Schema Analysis

Current SQLAlchemy model: `models/routine_feedback.py`.

`RoutineFeedback` columns:

- `id: String(36)`, primary key, UUID string.
- `user_id: String(128)`, nullable, indexed.
- `routine_draft_id: String(128)`, non-null, indexed.
- `rating: Integer`, nullable, validated by API as `1..5`.
- `completed: Boolean`, nullable.
- `accepted_without_edits: Boolean`, nullable.
- `skipped_exercise_ids: JSON`, nullable, stores `list[str]`.
- `edited_exercises: JSON`, nullable, stores `list[dict]`.
- `user_note: Text(500)`, nullable, stripped and stored but must not be used for ranking.
- `created_at: DateTime`, server default, indexed.
- `source: String(32)`, default `"api"`.

Current request schema: `engines.schemas.RoutineFeedbackRequest`.

- `routine_draft_id` is required and stripped.
- `user_id` is optional and stripped.
- `rating` is optional and constrained to `1..5`.
- `completed`, `accepted_without_edits` are optional booleans.
- `skipped_exercises` is a list of stripped non-empty exercise ID strings, max 128 chars.
- `edited_exercises` uses `EditedExercise`:
  - `original_exercise_id: str`
  - `replacement_exercise_id: Optional[str]`
  - `reason: Literal["too_heavy", "too_easy", "pain", "no_equipment", "dislike", "duplicate", "other"]`
- At least one feedback field must be present.

Important limitations:

- `routine_draft_id` is stored but there is no routine draft table or routine block snapshot available for reliable joins.
- Routine-level signals like `rating`, `completed`, and `accepted_without_edits` cannot be safely assigned to specific exercises unless exercise IDs are present in `skipped_exercise_ids` or `edited_exercises`.
- Candidate IDs in `candidate_ranker.py` are read as `str(candidate.get("id") or "")`; aggregation keys must preserve string exercise IDs.
- `candidate_ranker.py` currently scores candidates deterministically and returns dicts containing `score` and `score_reasons`.

## 3. Aggregation Scope

### User-Level Aggregation

User-level stats use only rows where `RoutineFeedback.user_id == request.user_id`.

Use when:

- `request.user_id` is present.
- A candidate exercise ID has at least one user-specific feedback event.

Priority:

- User-level adjustment should dominate global adjustment because it represents explicit personal preference.
- If user feedback conflicts with global feedback, apply the user adjustment plus a weaker global adjustment, then clamp.

### Global-Level Aggregation

Global stats use all feedback rows.

Use when:

- User-specific data is missing or low-confidence.
- Candidate has enough global feedback to provide weak cold-start signal.

Global adjustment must be weaker than user adjustment. Recommended multiplier: `0.3`.

### Routine-Level Aggregation

Routine-level signals should be stored in aggregate stats but should not strongly affect individual exercises in V1.

Reason:

- Current storage does not include a routine draft exercise list.
- Applying a low rating to every unseen exercise would create false negatives.

V1 policy:

- Count routine-level fields globally for observability.
- Apply routine-level positive/negative only to exercise IDs directly mentioned in `skipped_exercise_ids` or `edited_exercises`.
- Do not infer exercise-level feedback from `routine_draft_id` until routine content snapshots exist.

## 4. Signal Mapping

### Direct Exercise-Level Signals

Negative:

- `skipped_exercise_ids`: direct exercise-level negative.
- `edited_exercises.original_exercise_id`: direct exercise-level negative.
- `edited_exercises.reason == "pain"`: strong negative for original exercise.
- `edited_exercises.reason == "dislike"`: medium negative for original exercise.
- `edited_exercises.reason == "no_equipment"`: moderate context-dependent negative for original exercise.
- `edited_exercises.reason == "duplicate"`: moderate negative, primarily future diversity signal.
- `edited_exercises.reason in ("too_heavy", "too_easy")`: weak ranking negative; better suited for future weight prescription tuning.

Positive:

- `edited_exercises.replacement_exercise_id`: direct positive for replacement exercise.

### Routine-Level Signals

Positive:

- `accepted_without_edits == true`: routine-level positive.
- `completed == true`: routine-level weak positive.
- `rating >= 4`: routine-level positive.

Negative:

- `completed == false`: routine-level weak negative.
- `rating <= 2`: routine-level negative.

V1 caveat:

- Routine-level signals should not be spread to all exercises without a routine block snapshot.
- For stats, keep counts like `completed_count`, `accepted_without_edits_count`, `high_rating_count`, and `low_rating_count`.
- For scoring, only apply them when the same row also mentions an exercise ID. Example: if a row skips `pushup` and has rating `2`, the skipped exercise can receive both skip negative and weak low-rating negative.

### Explicit Non-Signals

Do not use:

- `user_note`
- raw feedback free text
- medical free text
- NLP-derived preference
- telemetry warning strings
- Rule Critic rationale text

## 5. Aggregation Model And Score Formula

### Data Model

Use a small dataclass in `engines/feedback_aggregation.py`:

```python
@dataclass
class ExerciseFeedbackStats:
    exercise_id: str
    user_id: str | None = None
    feedback_count: int = 0
    skip_count: int = 0
    original_replaced_count: int = 0
    replacement_chosen_count: int = 0
    pain_count: int = 0
    no_equipment_count: int = 0
    dislike_count: int = 0
    duplicate_count: int = 0
    too_heavy_count: int = 0
    too_easy_count: int = 0
    other_reason_count: int = 0
    completed_count: int = 0
    incomplete_count: int = 0
    accepted_without_edits_count: int = 0
    high_rating_count: int = 0
    low_rating_count: int = 0
    rating_sum: int = 0
    rating_count: int = 0
    last_feedback_at: datetime | None = None
```

Derived property:

```python
avg_rating = rating_sum / rating_count if rating_count else None
```

### Score Scale Consideration

Current `candidate_ranker.py` base score components include:

- mapped substitute: `+12`
- efficiency tier: up to `+70`
- compound: `+30`
- loadable equipment: `+18`
- primary target: `+20`
- large muscle: `+10`
- preferred exercise: `+20`
- bodyweight: `-18`
- DOMS moderate: `-25`
- recent repetition: `-6`

Typical candidate scores can vary by dozens of points. Feedback adjustment should be meaningful but unable to override hard constraints or dominate deterministic ranking. Recommended V1 clamp: `[-10, +10]`.

### Formula

Recommended weights:

```python
positive_score =
    1.0 * replacement_chosen_count
  + 0.25 * completed_count
  + 0.25 * accepted_without_edits_count
  + 0.25 * high_rating_count

negative_score =
    1.0 * skip_count
  + 1.0 * original_replaced_count
  + 1.5 * pain_count
  + 1.0 * dislike_count
  + 0.6 * no_equipment_count
  + 0.6 * duplicate_count
  + 0.3 * too_heavy_count
  + 0.3 * too_easy_count
  + 0.25 * low_rating_count
```

Confidence:

```python
confidence = min(1.0, feedback_count / 10.0)
```

Adjustment:

```python
raw_adjustment = (positive_score - negative_score) * confidence
feedback_adjustment = clamp(raw_adjustment * 3.0, -10.0, 10.0)
```

Reasoning:

- A single skip should not destroy ranking: `-1 * 0.1 * 3 = -0.3`.
- Ten repeated skips should reach meaningful penalty: `-10 * 1.0 * 3`, clamped to `-10`.
- Pain repeatedly selected as an edit reason reaches the lower clamp faster.
- Replacement-chosen signal can offset weak global negatives but not hard filters.

### Blending User And Global

Recommended V1:

```python
blended_adjustment = clamp(user_adjustment + 0.3 * global_adjustment, -10.0, 10.0)
```

If there is no user ID:

```python
blended_adjustment = clamp(0.3 * global_adjustment, -3.0, 3.0)
```

This keeps anonymous requests close to deterministic ranking.

## 6. DB / Query Strategy

### Option A: On-The-Fly Aggregation

Query `RoutineFeedback` during routine generation and aggregate in Python.

Pros:

- No migration required.
- Easy to test with existing in-memory DB fixtures.
- Good for P2-2 rollout behind a disabled feature flag.

Cons:

- JSON fields require Python-side parsing.
- Latency can grow with feedback table size.
- Need query bounds.

Recommended P2-2 V1 query policy:

- Only query when `ENABLE_FEEDBACK_AWARE_RANKING=true`.
- Only query for the candidate IDs already fetched from DB.
- Use `created_at >= now - FEEDBACK_AGGREGATION_LOOKBACK_DAYS`, default `180`.
- Use `limit`, default `5000`, to cap rows.
- User query: filter by `user_id`.
- Global query: no user filter, same lookback and limit.
- Python-side filtering should only count exercise IDs present in the current candidate ID set.

Suggested function signatures:

```python
def collect_feedback_stats(
    session: Session,
    *,
    user_id: str | None = None,
    exercise_ids: Iterable[str] | None = None,
    lookback_days: int = 180,
    limit: int = 5000,
) -> dict[str, ExerciseFeedbackStats]:
    ...

def compute_feedback_adjustment(
    stats: ExerciseFeedbackStats | None,
    *,
    multiplier: float = 3.0,
    clamp_min: float = -10.0,
    clamp_max: float = 10.0,
) -> float:
    ...

def get_feedback_adjustments(
    session: Session,
    *,
    user_id: str | None,
    exercise_ids: Iterable[str],
    global_weight: float = 0.3,
) -> dict[str, float]:
    ...

def blend_user_and_global_feedback(
    user_adjustment: float,
    global_adjustment: float,
    *,
    global_weight: float = 0.3,
    clamp_min: float = -10.0,
    clamp_max: float = 10.0,
) -> float:
    ...
```

### Option B: Aggregated Table

Create `exercise_feedback_stats` with precomputed counts.

Pros:

- Low generation-time latency.
- Easier dashboards and operational monitoring.
- Scales better once feedback volume grows.

Cons:

- Requires DB migration.
- Requires update path on feedback submission or a batch job.
- More consistency and backfill complexity.

Suggested future table:

- `id`
- `scope`: `"global"` or `"user"`
- `user_id`: nullable
- `exercise_id`
- count columns matching `ExerciseFeedbackStats`
- `last_feedback_at`
- `updated_at`

Recommendation:

- P2-2 should use Option A.
- Revisit Option B when feedback rows exceed a threshold such as 50k rows or query latency becomes measurable in staging.

## 7. Feature Flag / Config

Primary flag:

- `ENABLE_FEEDBACK_AWARE_RANKING=false`

Optional config:

- `FEEDBACK_RANKING_GLOBAL_WEIGHT=0.3`
- `FEEDBACK_RANKING_MAX_ADJUSTMENT=10`
- `FEEDBACK_AGGREGATION_LOOKBACK_DAYS=180`
- `FEEDBACK_AGGREGATION_MAX_ROWS=5000`

Resolver policy:

- Invalid booleans fall back to `false`.
- Invalid numeric values fall back to defaults.
- Feature flag off means no feedback query, no score changes, no latency overhead except import-level constants.

## 8. Candidate Ranker Integration Plan For P2-2

Current call chain:

- `routine_pipeline.generate_smart_routine()`
- `get_candidate_exercises()`
- `score_candidate_exercises(...)`
- prompt candidate payload from ranked candidates

Recommended P2-2 integration:

1. Add `engines/feedback_aggregation.py`.
2. Add `engines/feedback_ranking_policy.py` or place env resolvers in `feedback_aggregation.py`.
3. In `routine_pipeline.py`, after candidate DB fetch and before ranking:
   - collect candidate IDs as strings.
   - if flag enabled, compute `feedback_adjustments = get_feedback_adjustments(db, user_id=req.user_id, exercise_ids=ids)`.
   - pass adjustments to `score_candidate_exercises(..., feedback_adjustments=feedback_adjustments)`.
4. Extend `score_candidate_exercises()` with default parameter:

```python
feedback_adjustments: Optional[dict[str, float]] = None
```

5. Inside score loop, after existing deterministic score and before append:

```python
adjustment = float((feedback_adjustments or {}).get(str(candidate.get("id") or ""), 0.0))
if adjustment:
    score += adjustment
    reasons.append(f"feedback adjustment {adjustment:+.1f}")
```

6. Keep hard exclusions before feedback adjustment:
   - unavailable equipment
   - pain triggers
   - DOMS level 3
   - unpreferred exercise IDs

Do not:

- Introduce candidates not returned by DB.
- Override blocked equipment or pain filters.
- Change fallback or Rule Critic triggers.
- Change candidate pool default.
- Use `user_note`.

## 9. Privacy Policy

Hard privacy rules:

- `user_note` is excluded from aggregation.
- No free text is stored in stats.
- No raw feedback text is logged.
- Do not emit per-user preference telemetry in V1.
- Avoid logging `user_id` with adjustment maps.
- If logging is necessary, log only aggregate counts and candidate_count, never exercise-specific per-user details.

Allowed aggregate fields:

- exercise_id
- count fields
- avg rating numeric value
- last feedback timestamp
- computed adjustment

Sensitive/non-allowed fields:

- user_note
- raw request body
- free-text medical notes
- raw prompt / raw LLM output
- rationale text

## 10. Test Plan

Add `tests/test_feedback_aggregation.py`.

Required unit tests:

1. no feedback -> adjustment `0`.
2. skipped exercise -> negative adjustment.
3. replacement chosen -> positive adjustment.
4. reason `pain` -> stronger negative than generic edit.
5. reason `no_equipment` -> moderate negative.
6. reason `dislike` -> negative.
7. reason `duplicate` -> negative.
8. `accepted_without_edits` contributes only weakly and only when exercise is directly mentioned.
9. `completed=true` weak positive.
10. high rating weak positive.
11. low rating weak negative.
12. one feedback has low confidence.
13. ten feedback rows reach full confidence.
14. upper clamp applies.
15. lower clamp applies.
16. user-level stats are separate from global stats.
17. `user_id=None` uses global aggregation only.
18. user-specific adjustment dominates weak global adjustment.
19. `user_note` is ignored and never appears in stats repr or logs.
20. exercise IDs are preserved as strings.
21. malformed JSON/null JSON fields are skipped safely.
22. duplicate exercise IDs in one feedback row are counted once per signal type per row to avoid accidental overcounting.

Candidate ranker integration tests for P2-2:

1. flag off -> ranked list exactly identical to pre-feedback ranking.
2. flag on with no feedback -> ranking identical.
3. flag on with positive replacement signal -> candidate score increases.
4. flag on with skip/pain signals -> candidate score decreases.
5. feedback adjustment appears in `score_reasons`.
6. feedback does not reintroduce equipment-blocked candidate.
7. feedback does not reintroduce pain-trigger candidate.
8. feedback cannot override `unpreferred_exercise_ids` exclusion.
9. global adjustment applies at reduced weight.
10. anonymous request applies only weak global adjustment.

Regression commands for P2-2:

```bash
pytest tests/test_feedback_aggregation.py -q
pytest tests/test_routine_feedback.py -q
pytest tests/test_prompt_regression.py tests/test_guard.py tests/test_contract.py tests/test_fallback.py -q
pytest tests/evaluation -q
pytest tests/ -q
```

Evaluation plan:

- Extend offline evaluation helper to allow mock feedback adjustments.
- Compare flag-off and flag-on ranking for the same candidate set.
- Verify hard violation count remains zero.
- Verify Rule Critic score does not regress in fixture cases.
- Verify candidate payload growth is negligible because feedback only changes order/score reasons.

## 11. Risk Analysis

Risk: routine-level rating is misattributed to individual exercises.

- Mitigation: do not spread routine-level feedback without direct exercise IDs or routine snapshot.

Risk: global feedback suppresses valid exercises for users with different context.

- Mitigation: global weight `0.3`, anonymous clamp `[-3, +3]`, context-dependent reasons are weak.

Risk: feedback overrides hard safety filters.

- Mitigation: apply feedback only after existing hard exclusions.

Risk: query latency increases.

- Mitigation: flag off by default, candidate ID filter, lookback, max row limit.

Risk: user free text leaks into ranking or logs.

- Mitigation: never read `user_note` in aggregation; add tests asserting absence.

Risk: duplicate feedback submissions skew counts.

- Current API allows duplicate routine_draft_id submissions.
- V1 should accept duplicates as repeated user signals.
- Within a single row, deduplicate repeated exercise IDs per signal to avoid malformed payload amplification.

Risk: exercise ID mismatch.

- Candidate IDs may be numeric in DB rows but are consumed as strings in ranker.
- Normalize all feedback and candidate IDs with `str(value).strip()`.

## 12. Codex Implementation Prompt For P2-2

Implement feedback aggregation and feature-flagged candidate ranking integration.

Requirements:

1. Add `engines/feedback_aggregation.py`.
2. Implement `ExerciseFeedbackStats`.
3. Implement:
   - `resolve_feedback_aware_ranking_enabled(value: str | None = None) -> bool`
   - `get_feedback_aware_ranking_enabled() -> bool`
   - `collect_feedback_stats(session, user_id=None, exercise_ids=None, lookback_days=180, limit=5000)`
   - `compute_feedback_adjustment(stats, ...)`
   - `blend_user_and_global_feedback(user_adjustment, global_adjustment, ...)`
   - `get_feedback_adjustments(session, user_id, exercise_ids, ...)`
4. Feature flag name: `ENABLE_FEEDBACK_AWARE_RANKING`.
5. Default flag value: `false`.
6. Do not query feedback when flag is off.
7. Do not use `user_note`.
8. Preserve exercise IDs as strings.
9. Extend `score_candidate_exercises()` with optional `feedback_adjustments`.
10. Apply feedback adjustment only after existing hard exclusions.
11. Add `feedback adjustment +x.x` / `feedback adjustment -x.x` to `score_reasons` only when non-zero.
12. Integrate in `routine_pipeline.py` before candidate ranking.
13. If feedback query fails, swallow or safe-fallback to no adjustments; generation must not fail due to feedback aggregation.
14. Add `tests/test_feedback_aggregation.py` with the test plan above.
15. Keep all existing tests passing.

Non-goals:

- ML/LTR.
- user_note NLP.
- automatic exercise blocking.
- feedback-aware weight prescription.
- UI/admin dashboard.
- candidate pool 18 default promotion.
- Rule Critic rebuild/fallback trigger changes.
