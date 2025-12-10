import pytest

@pytest.fixture(scope="session")
def mock_openai(monkeypatch):
    class MockOpenAI:
        def __init__(self):
            pass
        def Completion_create(self, *args, **kwargs):
            return {"choices": [{"text": "mocked response"}]}
    monkeypatch.setattr("openai.Completion.create", MockOpenAI().Completion_create)
    yield

@pytest.fixture(scope="session")
def mock_azure(monkeypatch):
    class MockAzureML:
        def __init__(self):
            pass
        def run(self, *args, **kwargs):
            return {"result": "mocked azure response"}
    monkeypatch.setattr("azureml.core.Model.run", MockAzureML().run)
    yield