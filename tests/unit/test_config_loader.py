"""
Unit tests for config_loader — derived from src/utils/config_loader.py.

Tests cover:
- load_config: YAML parsing, missing file error
- get_config: caching behavior
- Edge cases
"""
import pytest
import yaml
import tempfile
import os
from pathlib import Path

from utils.config_loader import load_config, get_config, _config_cache


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Reset the module-level cache before each test."""
    import utils.config_loader as config_loader
    config_loader._config_cache = None
    yield
    config_loader._config_cache = None


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config.yaml file."""
    config_data = {
        'chunking': {
            'chunk_size': 350,
            'overlap': 50,
            'token_chunking': True,
        },
        'retrieval': {
            'top_k': 10,
            'semantic_weight': 0.6,
        },
        'paths': {
            'pdf_dir': 'protocolosPDF',
            'vectordb_dir': 'chroma_bge_m3',
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data), encoding='utf-8')
    return str(config_path)


class TestLoadConfig:
    def test_loads_valid_yaml(self, temp_config):
        config = load_config(temp_config)
        assert config['chunking']['chunk_size'] == 350
        assert config['retrieval']['top_k'] == 10
        assert config['paths']['pdf_dir'] == 'protocolosPDF'

    def test_missing_file_raises_filenotfound(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_loads_yaml_with_unicode(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("key: valor con acentos áéíóú", encoding='utf-8')
        config = load_config(str(config_path))
        assert "acentos" in config['key']

    def test_empty_yaml_returns_none(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("", encoding='utf-8')
        config = load_config(str(config_path))
        assert config is None


class TestGetConfig:
    def test_returns_config(self, temp_config):
        config = get_config(temp_config, use_cache=False)
        assert config['retrieval']['top_k'] == 10

    def test_caches_config(self, temp_config):
        # First call loads from file
        config1 = get_config(temp_config, use_cache=True)
        # Delete the file
        os.remove(temp_config)
        # Second call should return cached version (file is gone)
        config2 = get_config(temp_config, use_cache=True)
        assert config1 == config2

    def test_no_cache_reloads_from_file(self, temp_config):
        config1 = get_config(temp_config, use_cache=False)
        # Modify the file
        with open(temp_config, 'w', encoding='utf-8') as f:
            yaml.dump({'new_key': 'new_value'}, f)
        config2 = get_config(temp_config, use_cache=False)
        assert config2.get('new_key') == 'new_value'

    def test_missing_file_raises_filenotfound(self):
        with pytest.raises(FileNotFoundError):
            get_config("/nonexistent/config.yaml", use_cache=False)
