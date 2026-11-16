"""Local asynchronous jobs, shared preparation, isolated model training and activation."""
import copy
import csv
import json
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core import ROOT, file_rows, finite, load_config
from model_registry import resolve_model, valid_id

ACTIVE_STATES = {'preparing', 'training', 'cancelling'}


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def training_rows(payload):
    rows = file_rows(payload)
    if len(rows) > 100_000:
        raise ValueError('训练数据最多100000条')
    units = [payload.get(key, 'linear') for key in ('magnitude_unit', 'matched_magnitude_unit')]
    if any(unit not in ('linear', 'db') for unit in units):
        raise ValueError('训练前后幅值单位须分别选择linear或db')
    aliases = [('mag0', 'mag'), ('deg0', 'deg', 'meg0'), ('l1',), ('l2',),
               ('matched_mag', 'match_mag'), ('matched_deg', 'match_deg')]
    records = []
    for index, row in enumerate(rows, 2):
        source = {key.strip().lower(): value for key, value in row.items()}
        values = []
        for keys in aliases:
            matches = [finite(source[k], f'训练表第{index}行 {keys[0]}') for k in keys if k in source]
            if not matches:
                raise ValueError(f'训练表缺少 {keys[0]} 列；训练需要幅值、相位、L1、L2和匹配后幅值、相位六列')
            if any(value != matches[0] for value in matches[1:]):
                raise ValueError(f'训练表第{index}行 {keys[0]} 重复列数值不一致')
            values.append(matches[0])
        for offset, unit in zip((0, 4), units):
            value = values[offset]
            if (unit == 'db' and value > 0) or (unit == 'linear' and not 0 <= value <= 1):
                raise ValueError(f'训练表第{index}行幅值超出{unit}范围')
            values[offset] = 10**(value/20) if unit == 'db' else value
        records.append([*values, str(source.get('group_id', '')).strip()])
    if len(records) < 20:
        raise ValueError('训练数据至少20条，并须包含至少10个独立输入或实验组')
    return records


def parameters(payload, config):
    defaults = {'epochs': 150, 'surrogate_epochs': 100, 'patience': 25, 'width': 128,
                'blocks': 4, 'layers': 8, 'batch_size': 128, 'candidates': 32,
                'fourier_features': 32, 'fourier_scale': 1., 'surrogate_width': 96,
                'lr': .001, 'response_weight': .05, 'seed': 42, 'threads': 2,
                'device': config.get('device', 'cpu')}
    supplied = payload.get('parameters') or {}
    if set(supplied) - set(defaults):
        raise ValueError('存在未知训练参数')
    defaults.update(supplied)
    limits = {'epochs': (1, 10000), 'surrogate_epochs': (1, 10000), 'patience': (1, 10000),
              'width': (8, 2048), 'blocks': (1, 32), 'layers': (1, 32),
              'batch_size': (1, 8192), 'candidates': (1, 1024), 'fourier_features': (1, 512),
              'surrogate_width': (8, 2048), 'seed': (0, 2**31-1), 'threads': (1, 64)}
    for key, (low, high) in limits.items():
        value = finite(defaults[key], key)
        if not value.is_integer() or not low <= value <= high:
            raise ValueError(f'{key}须为{low}～{high}的整数')
        defaults[key] = int(value)
    for key in ('lr', 'fourier_scale', 'response_weight'):
        value = finite(defaults[key], key)
        if not (0 <= value <= 100) or (key != 'response_weight' and value == 0):
            raise ValueError(f'{key}超出有效范围')
        defaults[key] = value
    if defaults['device'] not in ('cpu', 'cuda'):
        raise ValueError('device仅支持cpu或cuda')
    return defaults


class TrainingService:
    def __init__(self, config_path, model_lock, root=None):
        self.config_path = Path(config_path).resolve()
        self.root = Path(root or ROOT / 'runs/training').resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_lock = model_lock
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.process = None
        self.thread = None
        self.job = None
        latest = self.root / 'latest.json'
        if latest.exists():
            pointer = json.loads(latest.read_text(encoding='utf-8'))
            path = (self.root / pointer['id'] / 'job.json').resolve()
            if path.is_relative_to(self.root) and path.exists():
                self.job = json.loads(path.read_text(encoding='utf-8'))
                if self.job['status'] in ACTIVE_STATES:
                    self.job.update(status='interrupted', error='服务上次运行被中断，请重新训练；未自动启用未完成权重。')
                    self._save()

    def _save(self):
        save_json(self.root / self.job['id'] / 'job.json', self.job)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.job) if self.job else {'status': 'idle', 'models': [], 'logs': []}

    def _update(self, **values):
        with self.lock:
            self.job.update(values)
            self._save()

    def start(self, payload):
        config = load_config(self.config_path)
        selected = payload.get('models')
        if not isinstance(selected, list) or not 1 <= len(selected) <= 16:
            raise ValueError('请选择1～16个模型配置')
        ids = [item.get('id') for item in selected]
        if len(set(ids)) != len(ids) or not all(valid_id(key) for key in ids):
            raise ValueError('模型标识须唯一且使用小写字母、数字和下划线')
        models = []
        for item in selected:
            model = {'id': item['id'], 'type': item['type'], 'label': str(item.get('label') or item['id'])[:80]}
            runtime = resolve_model(model, config)
            models.append({**model, 'runtime_dir': runtime['runtime_dir'], 'status': 'queued', 'progress': 0})
        args = parameters(payload, config)
        records = training_rows(payload)
        if not self.model_lock.acquire(blocking=False):
            raise ValueError('当前正在训练或推理，请等待完成后重试')
        try:
            job_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
            folder = self.root / job_id
            folder.mkdir()
            data_path = folder / 'prepared.npz'
            source = folder / 'training_linear.csv'
            with source.open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Mag0', 'Deg0', 'L1', 'L2', 'matched_mag', 'matched_deg', 'group_id'])
                writer.writerows(records)
            save_json(folder / 'data_config.json', {'magnitude_unit': 'linear', 'phase_unit': 'deg', 'length_unit': 'deg', 'seed': args['seed']})
            with self.lock:
                self.cancel_event.clear()
                self.job = {'id': job_id, 'status': 'preparing', 'filename': str(payload.get('filename', 'training.xlsx')),
                            'rows': len(records), 'models': models, 'parameters': args, 'logs': [], 'progress': 0,
                            'folder': str(folder), 'data': str(data_path), 'activated': False,
                            'created_at': datetime.now(timezone.utc).isoformat()}
                self._save()
                save_json(self.root / 'latest.json', {'id': job_id})
            self.thread = threading.Thread(target=self._run, args=(config,), daemon=True)
            self.thread.start()
            return self.snapshot()
        except BaseException:
            self.model_lock.release()
            raise

    def _spawn(self, request, model=None):
        folder = Path(self.job['folder'])
        request_path = folder / ('prepare_request.json' if model is None else model['id'] + '_request.json')
        save_json(request_path, request)
        flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
        with self.lock:
            if self.cancel_event.is_set():
                raise InterruptedError('用户已取消训练')
            self.process = subprocess.Popen([self.python, '-X', 'utf8', '-u', '-B', str(ROOT / 'scripts/training_worker.py'), '--request', str(request_path)],
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', **flags)
            process = self.process
        with (folder / 'training.log').open('a', encoding='utf-8') as log:
            for line in process.stdout:
                line = line.rstrip()
                log.write(line+'\n');log.flush()
                with self.lock:
                    self.job['logs'] = (self.job['logs'] + [line])[-80:]
                    match = re.search(r'(forward|inverse) epoch=(\d+)', line)
                    if model and match:
                        phase, epoch = match.group(1), int(match.group(2))
                        maximum = self.job['parameters']['surrogate_epochs' if phase == 'forward' else 'epochs']
                        progress = (.5 if phase == 'inverse' else 0) + .5*min(epoch/maximum, 1)
                        model.update(phase=phase, epoch=epoch, max_epochs=maximum, progress=progress)
                        self.job['progress'] = sum(item['progress'] for item in self.job['models']) / len(self.job['models'])
                    self._save()
        code = process.wait()
        with self.lock:
            self.process = None
        if self.cancel_event.is_set():
            raise InterruptedError('用户已取消训练')
        if code:
            raise ValueError('训练进程失败，请查看日志：'+'\n'.join(self.job['logs'][-8:]))

    def _run(self, config):
        try:
            self.python = config['python_executable']
            folder = Path(self.job['folder'])
            # Built-in preparation runs exactly once, all architectures receive this same archive.
            self._spawn({'mode': 'prepare', 'runtime_dir': str(ROOT / 'models/residual_mlp'),
                         'source': str(folder / 'training_linear.csv'), 'data': self.job['data'],
                         'config': str(folder / 'data_config.json'), 'summary': str(folder / 'data_summary.json')})
            summary = json.loads((folder / 'data_summary.json').read_text(encoding='utf-8'))
            self._update(status='training', data_summary=summary)
            for model in self.job['models']:
                if self.cancel_event.is_set():
                    raise InterruptedError('用户已取消训练')
                with self.lock:
                    model['status'] = 'training'
                    self._save()
                output = folder / model['id']
                args = {**self.job['parameters'], 'data': self.job['data'], 'output': str(output)}
                self._spawn({'mode': 'train', 'runtime_dir': model['runtime_dir'], 'args': args}, model)
                checkpoint = output / 'best.pt'
                if not checkpoint.is_file():
                    raise ValueError(f'{model["label"]}未生成best.pt')
                training_summary = json.loads((output / 'training_summary.json').read_text(encoding='utf-8'))
                with self.lock:
                    model.update(status='completed', progress=1, checkpoint=str(checkpoint), best_epoch=training_summary['best_epoch'])
                    self._save()
            with self.lock:
                if self.cancel_event.is_set():
                    raise InterruptedError('用户已取消训练')
                self._update(status='completed', progress=1, finished_at=datetime.now(timezone.utc).isoformat())
        except Exception as error:
            cancelled = isinstance(error, InterruptedError)
            with self.lock:
                for model in self.job['models']:
                    if model['status'] in ('queued', 'training'):
                        model['status'] = 'cancelled' if cancelled else 'failed'
            self._update(status='cancelled' if cancelled else 'failed', error=str(error))
        finally:
            self.model_lock.release()

    def cancel(self):
        with self.lock:
            if not self.job or self.job['status'] not in ACTIVE_STATES:
                raise ValueError('没有正在运行的训练任务')
            self.cancel_event.set()
            self.job['status'] = 'cancelling'
            if self.process and self.process.poll() is None:
                self.process.terminate()
            self._save()
            return self.snapshot()

    def shutdown(self):
        if self.snapshot()['status'] in ACTIVE_STATES:
            try:
                self.cancel()
            except ValueError:
                pass
        if self.thread:
            self.thread.join(timeout=10)

    def activate(self, job_id):
        if not self.model_lock.acquire(blocking=False):
            raise ValueError('当前正在训练或推理，请稍后启用权重')
        try:
            with self.lock:
                if not self.job or self.job['id'] != job_id or self.job['status'] != 'completed':
                    raise ValueError('只能启用当前已完整训练成功的任务')
                models = [{key: item[key] for key in ('id', 'type', 'label', 'checkpoint')} | {'prepared_data': self.job['data']} for item in self.job['models']]
                if not all(Path(item['checkpoint']).is_file() for item in models):
                    raise ValueError('训练权重文件已缺失')
                active = self.config_path.with_name(self.config_path.stem + '.active.local.json')
                save_json(active, {'models': models, 'evaluation_model': models[0]['id']})
                self.job['activated'] = True
                self._save()
                return self.snapshot()
        finally:
            self.model_lock.release()
