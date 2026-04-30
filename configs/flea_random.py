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
import os
config['io'] = {
    'reward': {
        'type': 'ISMATEC_SERIAL',
        'address': os.environ.get('PUMP', '/dev/ttyACM0'),
        'channels': [
            {'channel': '1', 'clockwise': True, 'speed': 100},
            {'channel': '4', 'clockwise': True, 'speed': 100}
        ]
    }
}

SIZE = 1920, 1080
orientation = 'portrait'
# orientation = 'landscape'
if orientation == 'portrait':
    SIZE = SIZE[1], SIZE[0]

CENTER = SIZE[0]//2, SIZE[1]//2
X_OFFSET = 200
Y_OFFSET = 200
# DIAGONAL
LEFT = (CENTER[0]-X_OFFSET, CENTER[1]+Y_OFFSET)
RIGHT = (CENTER[0]+X_OFFSET, CENTER[1]-Y_OFFSET)

RADIUS = 400
N_POSITIONS = 6
START_ANGLE = 0
angle_offsets = START_ANGLE + (2*np.pi / N_POSITIONS) * np.arange(N_POSITIONS)
X, Y = RADIUS*np.cos(angle_offsets), RADIUS*np.sin(angle_offsets)

config['locations'] = {
    'center': CENTER,
}
for i, (x, y) in enumerate(zip(X, Y)):
    config[i] = (x, y)

config['display'] = {
  'size': SIZE,
  'display': 0,
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
    'show': False,
    'template_path': 'server',
}

magnitudes = range(1, 6)
reward_duration = 0.4  # in seconds
interpulse_interval = 0.2  # in seconds
config['reward_channels'] = ('1','4')
config['magnitude_mapping'] = {
    mag: {'duration': reward_duration*mag, 'n_pulses': 1, 'interpulse_interval': 0}
    for mag in magnitudes
}
# # stimulus set 1
# images = [f'stimuli/aada{chr(97+i)}.png' for i in range(5)]
# # stimulus set 2
# images = [f'stimuli/aada{chr(97+5+i)}.png' for i in range(5)]
# stimulus set 3
images = [f'stimuli/aada{chr(97+10+i)}.png' for i in range(5)]
config['items'] = dict(zip(magnitudes, images))

forced_choice_trials = {
    f'f{mag}{loc}': dict(
        magnitude=mag,
        loc=loc,
        trial_type='forced'
    )
    for mag, loc in product(magnitudes, ['left', 'right'])
}

locations = range(N_POSITIONS)
location_pairs = [(i, (i+(N_POSITIONS//2))%N_POSITIONS) for i in locations]

two_afc_trials = {
    i: dict(
        magnitudes=(option1, option2),
        locs=locs,
        trial_type='choice'
    )
    for i, ((option1, option2), locs) in enumerate(product(combinations(magnitudes, 2), location_pairs))
}

config['conditions'] = {**forced_choice_trials, **two_afc_trials}

# design the block structure
blocks = {}
# we will have blocks of 2AFC trials mixing all magnitude levels, starting with easy trials
for value_difference in range(4, 0, -1):
    condition_list = []
    for condition, info in two_afc_trials.items():
        op1, op2 = info.get('magnitudes')
        if abs(op1-op2)==value_difference:
            condition_list.append(condition)

    current_block = f'valuediff{value_difference:d}'
    next_block = f'valuediff{max(value_difference-1, 1):d}'
    # previous_block = f'valuediff{min(value_difference+1, 4):d}'
    if value_difference == 4:
        previous_block = 0 if 0 in blocks else 'valuediff4'
    else:
        previous_block = f'valuediff{value_difference+1:d}'
    blocks[current_block] = dict(
        conditions=condition_list,
        length=10,
        retry={'timeout': True},
        transition=[
            {'condition': {'outcome': 'correct', 'min': 8}, 'next': next_block},
            {'condition': {'outcome': 'correct', 'min': 6}, 'next': current_block},
            {'next': previous_block}
        ],
        method='random'
    )

config['blocks'] = blocks

config['trial_types'] = {
    'forced': {
        'module': 'trials/forced.py',
        'class': 'ForcedChoiceTrial'
    },
    'choice': {
        'module': 'trials/twoafc.py',
        'class': 'TwoAFCTrial'
    }
}
config['hotkeys'] = {
    '4': {'do':'pump_on'},
    '8': {'do': 'pump_off'},
    '3': {'do': 'pause'},
    '7': {'do': 'unpause'},
    '5': {'do': 'quit'}
}

if __name__ == '__main__':
    import pprint
    pprint.pprint(config)
