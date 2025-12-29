from experiment.engine.pygame import PygameManager
from pathlib import Path
from collections import ChainMap


DEFAULT_DISPLAY_SETTINGS = {
    'size': (1080, 1920),
    'fullscreen': True,
    'display': 0
}
DEFAULT_REMOTE_SERVER = {
    'enabled': True,
    'show': False,
    'template_path': Path('server').absolute(),
}
DEFAULT_IO_SETTINGS = {
    'reward': {
        'type': 'ISMATEC_SERIAL',
        'address': '/dev/ttyACM0',
        'channels': [
            {'channel': '2', 'clockwise': True, 'speed': 100},
            {'channel': '3', 'clockwise': True, 'speed': 100}
        ]
    }
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

    config['strict_mode'] = config.get('strict_mode', True)
    config['display'] = display_settings
    config['background'] = background
    config['remote_server'] = remote_server_config
    config['io'] = config.get('io', DEFAULT_IO_SETTINGS)
    config['valid_times'] = config.get('valid_times', [
        {'start': '08:00', 'end': '18:00'},
    ])

    mgr = PygameManager(
        data_directory=data_directory,
        config=config
    )
    return mgr