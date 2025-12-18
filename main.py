from manager import load_manager

from experiment.blockmanager import BlockManager
from experiment.util.config import load_config

def main(config, **overrides):
    cfg = load_config(config)
    cfg.update(overrides)
    mgr = load_manager(cfg)
    mgr.run_session(BlockManager.from_config(cfg))

if __name__ == "__main__":
    import sys
    config = sys.argv[1]
    overrides = {}
    if '--debug' in sys.argv:
        overrides['strict_mode'] = False
    elif '--strict' in sys.argv:
        overrides['strict_mode'] = True

    main(config, **overrides)
