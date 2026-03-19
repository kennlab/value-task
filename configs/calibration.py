import numpy as np
from typing import Dict, Any
from itertools import product

config: Dict[str, Any] = dict(
    name='fleabottom',
    duration=10,
    size=(200,200),
    bbox=dict(width=300, height=300),
    allow_outside_touch=True,
    ITI=1.5,
    cue_incorrect=True,
    reward_feedback_method='bar_height'
)

config['io'] = {
    'reward': {
        'type': 'ISMATEC_SERIAL',
        'address': 'COM5',
        'channels': [
            {'channel': '1', 'clockwise': True, 'speed': 100},
            {'channel': '4', 'clockwise': True, 'speed': 100}
        ]
    }
}

SIZE = 1366, 768
orientation = 'portrait'
# orientation = 'landscape'
if orientation == 'portrait':
    SIZE = SIZE[1], SIZE[0]

CENTER = SIZE[0]//2, SIZE[1]//2

config['locations'] = {
    'center': CENTER,
}
gap = 300
for i in range(5):
    config['locations'][i] = CENTER[0], CENTER[1] + (i-2)*gap

config['display'] = {
  'size': SIZE,
  'display': 1,
  'fullscreen': True
}

config['debug'] = {
    'display': {
    'size': SIZE,
    'display': 0,
    'fullscreen': False
    }
}

config['remote_server'] = {
    'enabled': True,
    'show': True,
    'template_path': 'server',
}

magnitudes = range(1, 6)
reward_duration = 0.6  # in seconds
interpulse_interval = 0.2  # in seconds
config['reward_channels'] = ('1','4')
config['magnitude_mapping'] = {
    mag: {'duration': reward_duration*mag, 'n_pulses': 1, 'interpulse_interval': interpulse_interval}
    for mag in magnitudes
}
# # stimulus set 1
# images = [f'stimuli/aada{chr(97+i)}.png' for i in range(5)]
# # stimulus set 2
# images = [f'stimuli/aada{chr(97+5+i)}.png' for i in range(5)]
# stimulus set 3
images = [f'stimuli/aada{chr(97+10+i)}.png' for i in range(5)]
config['items'] = dict(zip(magnitudes, images))

config['conditions'] = {
    0: dict(
        magnitudes=magnitudes,
        locs=list(range(5)),
        trial_type='calibration'
    )
}

config['blocks'] = {
    0: {
        'conditions': [0],
        'length': 20
    }
}

config['trial_types'] = {
    'calibration': {
        'module': 'trials/calibration.py',
        'class': 'CalibrationTrial'
    }
}

if __name__ == '__main__':
    import pprint
    pprint.pprint(config)