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
# OFFSET = 250
# V_OFFSET = -200
# LEFT = (CENTER[0] - OFFSET, CENTER[1]-V_OFFSET)
# RIGHT = (CENTER[0] + OFFSET, CENTER[1]-V_OFFSET)
OFFSET = 300
V_OFFSET = 200
LEFT = (CENTER[0]+V_OFFSET, CENTER[1]-OFFSET)
RIGHT = (CENTER[0]+V_OFFSET, CENTER[1]+OFFSET)
config['locations'] = {
    'left': LEFT,
    'right': RIGHT,
    'center': CENTER,
}

config['display'] = {
  'size': SIZE,
#   'display': 1,
#   'fullscreen': True
   'display': 0,
   'fullscreen': False
}

config['remote_server'] = {
    'enabled': True,
    'show': True,
    'template_path': 'server',
}

magnitudes = range(1, 6)
reward_duration = 0.5  # in seconds
interpulse_interval = 0.2  # in seconds
config['reward_channels'] = ('1','4')
config['magnitude_mapping'] = {
    mag: {
        'n_pulses': mag, 
        'duration': reward_duration, 
        'interpulse_interval': interpulse_interval
    } 
    for mag in magnitudes
}

# stimulus set 1
images = [f'stimuli/aada{chr(97+i)}.png' for i in range(5)]
# stimulus set 2
images = [f'stimuli/aada{chr(97+5+i)}.png' for i in range(5)]
config['items'] = dict(zip(magnitudes, images))

forced_choice_trials = {
    f'f{mag}{loc}': dict(
        magnitude=mag,
        loc=loc,
        trial_type='forced'
    )
    for mag, loc in product(magnitudes, ['left', 'right'])
}
two_afc_trials = {
    f'c{option1}v{option2}_opt1{locs[0]}': dict(
        magnitudes=(option1, option2),
        locs=locs,
        trial_type='choice'
    )
    for (option1, option2), locs in product(product(magnitudes, repeat=2), [('left', 'right'), ('right', 'left')])
    if option1 != option2
}

config['conditions'] = {**forced_choice_trials, **two_afc_trials}

# design the block structure
## first we will have blocks of forced choice trials in 10 block trials at a given magnitude level
blocks = {}
# magnitudes_custom_order = [3, 1, 5, 2, 4]
# for i, mag in enumerate(magnitudes_custom_order):
#     blocks[i] = dict(
#         conditions=[f'f{mag}left', f'f{mag}right'],
#         length=1,
#         retry={'timeout': True},
#         transition=[
#             {'next': i+1}
#         ]
#     )
# # then one block mixing all forced choice trials
# blocks[len(blocks)] = dict(
#     conditions=[f'f{mag}left' for mag in magnitudes] + [f'f{mag}right' for mag in magnitudes],
#     length=10,
#     retry={'timeout': True},
#     transition=[
#         {'next': 'valuediff4'}
#     ]
# )
# then we will have blocks of 2AFC trials mixing all magnitude levels, starting with easy trials
for value_difference in range(4, 0, -1):
    condition_list = []
    for option1 in magnitudes:
        option2 = option1 - value_difference
        if option2 in magnitudes:
            condition_list.append(f'c{option1}v{option2}_opt1left')
            condition_list.append(f'c{option1}v{option2}_opt1right')

    current_block = f'valuediff{value_difference:d}'
    next_block = f'valuediff{max(value_difference-1, 1):d}'
    # previous_block = f'valuediff{min(value_difference+1, 4):d}'
    if value_difference == 4:
        previous_block = 0 if 0 in blocks else 'valuediff4'
    else:
        previous_block = f'valuediff{value_difference+1:d}'
    blocks[current_block] = dict(
        conditions=condition_list,
        length=20,
        retry={'timeout': True},
        transition=[
            {'condition': {'outcome': 'correct', 'min': 15}, 'next': next_block},
            {'condition': {'outcome': 'correct', 'min': 12}, 'next': current_block},
            {'next': previous_block}
        ]
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

if __name__ == '__main__':
    import pprint
    pprint.pprint(config)