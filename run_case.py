#!/usr/bin/env python3
"""List/resolve a paper case or dispatch one explicit workflow."""
import argparse
import importlib
import json
from interface.config import registry, resolve
from interface.outputs import Run


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('problem', nargs='?', choices=['l63','ks','kinetic'])
    p.add_argument('--list-paper-cases', action='store_true');p.add_argument('--json', action='store_true')
    p.add_argument('--paper-case');p.add_argument('--config');p.add_argument('--resolve-only', action='store_true')
    p.add_argument('--workflow', choices=['reproduce','replay','render'])
    p.add_argument('--render-kind', choices=['summary','validation_heatmap','rollout'])
    p.add_argument('--data-root');p.add_argument('--output');p.add_argument('--device',type=int)
    p.add_argument('--training-size',type=int);p.add_argument('--kernel',choices=['rbf','diffusion_maps'])
    p.add_argument('--stage-reconstruction');p.add_argument('--memory-order',type=int)
    a=p.parse_args(argv)
    if a.list_paper_cases:
        cases=registry(a.problem)
        print(json.dumps(cases,indent=2) if a.json else '\n'.join(f'{c["id"]}\t{c["backend"]}\t{c["readiness"]}' for c in cases))
        return 0
    if not a.problem: p.error('Choose a problem or --list-paper-cases')
    try:
        config=resolve(a.problem,paper_case=a.paper_case,config=a.config,
                       overrides={k:getattr(a,k) for k in ['training_size','kernel','stage_reconstruction','memory_order'] if getattr(a,k) is not None},
                       operational={k:getattr(a,k) for k in ['workflow','data_root','output','device','render_kind']})
    except (ValueError,KeyError,TypeError) as exc: p.error(str(exc))
    if a.resolve_only:
        print(json.dumps(config,indent=2));return 0
    run=Run(config)
    try:
        importlib.import_module(f'problems.{a.problem}.pipeline').run(config,run)
    except (ValueError,RuntimeError,FileNotFoundError,ImportError) as exc:
        run.finish('blocked_or_failed',str(exc));print(f'{exc}\nRun record: {run.path}');return 2
    except Exception as exc:
        run.finish('failed',f'{type(exc).__name__}: {exc}');raise
    run.finish('complete');print(run.path);return 0

if __name__ == '__main__':
    raise SystemExit(main())
