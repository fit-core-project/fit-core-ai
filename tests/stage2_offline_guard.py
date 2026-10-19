"""Portable oracle plugin: fixtures may not contact any TCP/UDP service."""
import socket

import pytest


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    original = socket.socket.connect

    def guarded(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("Portable core oracle forbids network access")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
