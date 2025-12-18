from experiment.engine.pygame import PygameManager
from pathlib import Path
from collections import ChainMap


DEFAULT_DISPLAY_SETTINGS = {
    'size': (1366, 768),
    'fullscreen': False,
    'display': 0
}
DEFAULT_REMOTE_SERVER = {
    'enabled': True,
    'show': True,
    'template_path': Path('server').absolute(),
}
def load_manager(config):
    monkey = config.get('name', None)
    data_directory=Path('data')
    if monkey is not None:
        data_directory = data_directory / monkey
    data_directory.mkdir(parents=True, exist_ok=True)

    ## Display settings
    display_settings = dict(ChainMap(config.get('display', {}), DEFAULT_DISPLAY_SETTINGS))
    background = config.get('background', (200, 200, 200))


    ## Remote server settings
    remote_server_settings = config.get('remote_server', {})
    remote_server_config = dict(ChainMap(remote_server_settings, DEFAULT_REMOTE_SERVER))

    strict_mode = config.get('strict_mode', True)

    mgr = PygameManager(
        data_directory=data_directory,
        config={
            'strict_mode': strict_mode,
            'display': display_settings,
            'background': background,
            'remote_server': remote_server_config,
            'io': {
                'reward': {
                    'type': 'ISMATEC_SERIAL',
                    'address': 'COM5',
                    'channels': [
                        {'channel': '2', 'clockwise': True, 'speed': 100},
                        {'channel': '3', 'clockwise': True, 'speed': 100}
                    ]
                }
            },
            'valid_times': [
                {'start': '09:00', 'end': '12:00'},
                {'start': '13:00', 'end': '17:00'}
            ]
        }
    )
    return mgr