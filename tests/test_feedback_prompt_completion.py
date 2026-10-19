import pytest

from engines.prompt_builder import build_compiled_system_prompt, build_system_prompt


@pytest.mark.parametrize("constrained", [False, True])
def test_compiled_selection_prompt_has_no_conflicting_request_for_prose(constrained):
    prompt = build_compiled_system_prompt(
        None, None, has_doms=constrained, has_medical_context=constrained,
        readiness_level="low" if constrained else "normal", training_style="gym",
        session_structure="superset" if constrained else "standard", has_adjunct_options=False,
    )
    assert "Use this context to write" not in prompt
    assert "omit that slot and explain" not in prompt
    assert "In Korean rationale, make" not in prompt
    assert "The server writes all user-facing" in prompt
    assert "Each selection must include only: slotId and exerciseId" in prompt
    assert "Language guidance below describes server-owned output" in prompt
    assert "Beginner users:" in prompt
    assert "This product coaches intermediate-and-above lifters" not in prompt


def test_prompt_keeps_health_and_slot_constraints_when_prose_ownership_is_clarified():
    prompt = build_system_prompt(None, None)
    assert "Treat DOMS level 3 as an exclusion constraint" in prompt
    assert "same slot bundle" in prompt
    assert "Never reinterpret managedNoRestrictions or recovered as an active injury" in prompt
