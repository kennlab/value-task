from manager import load_manager

from experiment.util.config import load_config

def main(config, **overrides):
    cfg = load_config(config)
    cfg.update(overrides)
    mgr = load_manager(cfg)
    mgr.run_session_from_config(cfg)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    config = sys.argv[1]
    config_path = Path(config)
    if not config_path.exists():
        config_paths = list(Path('configs').glob(f'{config}*'))
        if len(config_paths) == 1:
            config = config_paths[0].as_posix()
        else:
            raise ValueError(f'Could not find config {config}')
        
    overrides = {}
    if '--debug' in sys.argv:
        overrides['strict_mode'] = False
        overrides['display'] = {'fullscreen': False}
    elif '--strict' in sys.argv:
        overrides['strict_mode'] = True

    main(config, **overrides)