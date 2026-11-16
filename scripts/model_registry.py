"""Local, trusted architecture registry. Browser requests only select registered keys."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILTINS = {
    'residual_mlp': ('残差 MLP', 'models/residual_mlp'),
    'fourier_resmlp': ('Fourier ResMLP', 'models/fourier_resmlp'),
    'conditional_flow': ('条件 Flow', 'models/conditional_flow'),
}


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,63}', value) is not None


def registry(config):
    result = {key: {'id': key, 'label': label, 'runtime_dir': str(ROOT / folder)}
              for key, (label, folder) in BUILTINS.items()}
    for item in config.get('model_types', []):
        key = item['id']
        if not valid_id(key) or key in result:
            raise ValueError('自定义模型类型标识无效或重复')
        folder = Path(item['runtime_dir']).expanduser()
        folder = (ROOT / folder).resolve() if not folder.is_absolute() else folder.resolve()
        if not all((folder / name).is_file() for name in ('data.py', 'engine.py', 'model.py')):
            raise ValueError(f'自定义模型目录缺少 data.py / engine.py / model.py：{key}')
        result[key] = {'id': key, 'label': str(item.get('label', key)), 'runtime_dir': str(folder)}
    return result


def resolve_model(model, config):
    types = registry(config)
    kind = model.get('type', model['id'])
    if kind not in types:
        raise ValueError(f'未注册模型类型：{kind}')
    return types[kind]
