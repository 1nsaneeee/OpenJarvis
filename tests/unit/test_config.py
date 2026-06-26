import pytest

from openjarvis.system.config import AppConfig, load_config


def test_load_defaults(tmp_path):
    """load_config with a minimal yaml returns valid AppConfig."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("llm:\n  provider: openai\n  model: gpt-4o\n")
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, AppConfig)
    assert cfg.llm.provider == "openai"
    assert cfg.audio.sample_rate == 16000   # default


def test_env_override(tmp_path, monkeypatch):
    """REDIS_URL env var overrides config file value."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("")
    monkeypatch.setenv("REDIS_URL", "redis://myhost:6380/2")
    cfg = load_config(str(cfg_file))
    assert cfg.redis_url == "redis://myhost:6380/2"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nonexistent.yaml"))
