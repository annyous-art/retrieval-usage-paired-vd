"""Load the server's existing old_baseline client by absolute path.
No network requests occur during loading. Model-specific routes stay in that file.
"""
import importlib.util
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]


def local_config():
    path = Path(__file__).with_name('gateway_config_local.py')
    return load_file(path, 'slicerag_private_gateway') if path.exists() else None


def gateway_url():
    config = local_config()
    value = os.environ.get('SLICERAG_BASE_URL', getattr(config, 'BASE_URL', '')).strip().rstrip('/')
    if value:
        parsed = urlsplit(value)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('SLICERAG_BASE_URL must be an HTTPS API base URL without credentials, query or fragment')
        if not parsed.path.endswith('/v1'):
            raise ValueError('Use the API base URL ending in /v1, not /chat/completions or the documentation URL')
        if parsed.hostname == 'docs.gateway.example':
            raise ValueError('The documentation host is not your API gateway')
    return value


def baseline_dir():
    override = os.environ.get('SLICERAG_BASELINE_DIR')
    choices = [Path(override).expanduser().resolve()] if override else [ROOT / 'baseline', ROOT / 'old_baseline']
    for directory in choices:
        if all((directory / name).is_file() for name in ('model_api_clients.py', 'utils.py')):
            return directory
    raise FileNotFoundError('Expected baseline/{model_api_clients.py,utils.py}; set SLICERAG_BASELINE_DIR to the existing baseline directory.')


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_client():
    base_url = gateway_url()
    config = local_config()
    configured_host = getattr(config, 'BASE_URL', '').rstrip('/')
    gateway_key = os.environ.get('SLICERAG_API_KEY') or (getattr(config, 'API_KEY', '') if base_url == configured_host else '')
    if base_url and not gateway_key:
        raise ValueError('Configure a key for this gateway; credentials are not reused across hosts')
    if base_url:
        from gateway_client import GatewayClient
        return GatewayClient(base_url, gateway_key)
    path = baseline_dir() / 'model_api_clients.py'
    api = load_file(path, 'slicerag_legacy_api')
    key = os.environ.get('SLICERAG_API_KEY') or os.environ.get('OPENAI_API_KEY') or getattr(api, 'API_KEY', '')
    if not key or key == 'sk-':
        raise ValueError('Set SLICERAG_API_KEY, or configure the existing old_baseline API_KEY on your server.')
    # Keep the original gateway/model routing. Only replace credentials, if supplied.
    if key != api.API_KEY:
        api.API_KEY = key
        api.client = api.OpenAI(api_key=key, base_url=str(api.client.base_url))
        api.claude_client = api.Anthropic(api_key=key, base_url=str(api.claude_client.base_url))
    return api


def baseline_templates():
    return load_file(baseline_dir() / 'utils.py', 'slicerag_legacy_templates')
