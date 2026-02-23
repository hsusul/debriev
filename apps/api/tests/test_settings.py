from __future__ import annotations

from pathlib import Path

from app.settings import Settings


def test_settings_loads_values_from_env_file(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "DATABASE_URL=sqlite:///./from-env-file.db\nCOURTLISTENER_TOKEN=token-from-file\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("COURTLISTENER_TOKEN", raising=False)

    settings = Settings(_env_file=str(env_file))

    assert settings.database_url == "sqlite:///./from-env-file.db"
    assert settings.courtlistener_token == "token-from-file"
