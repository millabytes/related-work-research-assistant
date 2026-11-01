import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from related_work_assistant.api.main import app
from related_work_assistant.cli import app as cli
from related_work_assistant.core.config import PROJECT_ROOT, Settings


def test_settings(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss:/?#%")
    settings = Settings(_env_file=None)
    assert settings.sqlalchemy_url.password == "p@ss:/?#%"
    assert "p@ss" not in repr(settings)
    assert "p@ss" not in str(settings.sqlalchemy_url)
    assert settings.raw_pdfs_dir == PROJECT_ROOT / "data/raw_data/papers"


def test_url_secret_and_validation():
    settings = Settings(_env_file=None, database_url="postgresql+psycopg://u:secret@localhost/db")
    assert "secret" not in repr(settings)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="sqlite:///db")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, postgres_password="test", postgres_port=0)


def test_paths_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, postgres_password="test")
    assert settings.raw_pdfs_dir == PROJECT_ROOT / "data/raw_data/papers"


def test_health_without_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "invalid")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 503


def test_cli_help_and_failure(monkeypatch):
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0
    assert "regenerate-qwen" in help_result.output
    assert "qwen-status" in help_result.output
    assert "train-qwen-lora-smoke" in help_result.output
    assert "train-qwen-lora" in help_result.output
    assert "qwen-lora-training-status" in help_result.output
    assert "regenerate-qwen-lora" in help_result.output
    assert runner.invoke(cli, ["regenerate-qwen"]).exit_code != 0
    monkeypatch.setenv("DATABASE_URL", "invalid-secret")
    result = runner.invoke(cli, ["db-check"])
    assert result.exit_code == 1
    assert "invalid-secret" not in result.output
