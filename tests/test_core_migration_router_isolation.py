"""Routine model defaults must not change collaboration quicklog behavior."""
from unittest.mock import MagicMock

import pytest

from engines.llm_router import get_llm


@pytest.mark.parametrize('engine', ['routine', 'diet', 'nlp', 'supplement'])
def test_reasoning_default_is_routine_only(monkeypatch, engine):
    monkeypatch.setenv('LLM_PROVIDER', 'local')
    monkeypatch.delenv('APP_ENV', raising=False)
    monkeypatch.delenv('LOCAL_LLM_REASONING', raising=False)
    factory = MagicMock()
    monkeypatch.setattr('langchain_ollama.ChatOllama', factory)
    get_llm(engine)
    options = factory.call_args.kwargs
    if engine == 'routine':
        assert options['reasoning'] is False
    else:
        assert 'reasoning' not in options
    assert options.get('format') == (None if engine == 'supplement' else 'json')


@pytest.mark.parametrize('engine', ['routine', 'diet', 'nlp', 'supplement'])
def test_explicit_reasoning_environment_remains_authoritative(monkeypatch, engine):
    monkeypatch.setenv('LLM_PROVIDER', 'local')
    monkeypatch.delenv('APP_ENV', raising=False)
    monkeypatch.setenv('LOCAL_LLM_REASONING', 'true')
    factory = MagicMock()
    monkeypatch.setattr('langchain_ollama.ChatOllama', factory)
    get_llm(engine)
    assert factory.call_args.kwargs['reasoning'] is True


def test_windows_probe_needs_explicit_paths(monkeypatch, tmp_path):
    from scripts.check_local_llm_readiness import detect_windows_ollama_install

    monkeypatch.delenv('WINDOWS_OLLAMA_DATA_DIR', raising=False)
    monkeypatch.delenv('WINDOWS_OLLAMA_PROGRAM_DIR', raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'ollama.exe').touch()
    assert detect_windows_ollama_install()['windows_install_detected'] is False
    report = detect_windows_ollama_install(program_dir=str(tmp_path))
    assert report['windows_executable_detected'] is True
