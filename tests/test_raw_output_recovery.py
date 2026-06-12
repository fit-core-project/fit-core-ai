from __future__ import annotations

from engines.llm_parser import extract_raw_llm_output_from_exception


class CustomError(Exception):
    pass


def test_extract_raw_output_reads_llm_output():
    exc = CustomError("wrapper")
    exc.llm_output = '{"ok": true}'

    text, source = extract_raw_llm_output_from_exception(exc)

    assert text == '{"ok": true}'
    assert source == "exception_llm_output"


def test_extract_raw_output_reads_observation():
    exc = CustomError("wrapper")
    exc.observation = '{"observed": true}'

    text, source = extract_raw_llm_output_from_exception(exc)

    assert text == '{"observed": true}'
    assert source == "exception_observation"


def test_extract_raw_output_reads_string_args():
    exc = CustomError('{"from_args": true}')

    text, source = extract_raw_llm_output_from_exception(exc)

    assert text == '{"from_args": true}'
    assert source == "exception_args"


def test_extract_raw_output_walks_cause():
    cause = CustomError("cause")
    cause.llm_output = '{"from_cause": true}'
    exc = CustomError()
    exc.__cause__ = cause

    text, source = extract_raw_llm_output_from_exception(exc)

    assert text == '{"from_cause": true}'
    assert source == "exception_llm_output"


def test_extract_raw_output_ignores_empty_strings():
    exc = CustomError("")
    exc.llm_output = "   "

    text, source = extract_raw_llm_output_from_exception(exc)

    assert text is None
    assert source == "none"
