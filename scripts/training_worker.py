"""One isolated trusted runtime per preparation/training subprocess."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    sys.path.insert(0, request['runtime_dir'])
    if request['mode'] == 'prepare':
        from data import prepare, load
        prepare(request['source'], request['data'], request['config'])
        arrays, metadata = load(request['data'])
        summary = {'rows': metadata['rows'], 'independent_groups': metadata['independent_groups'],
                   'split_rows': {key: len(arrays[key + '_idx']) for key in ('train', 'val', 'test')},
                   'length_bounds': metadata['length_bounds']}
        Path(request['summary']).write_text(json.dumps(summary), encoding='utf-8')
    else:
        from engine import train
        train(SimpleNamespace(**request['args']))


if __name__ == '__main__':
    main()
