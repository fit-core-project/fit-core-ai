def pytest_configure(config):
    config.addinivalue_line("markers", "eval: offline routine generation evaluation scenario")
