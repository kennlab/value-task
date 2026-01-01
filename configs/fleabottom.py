import numpy as np
from typing import Dict, Any
from itertools import product

config: Dict[str, Any] = dict(
    name='fleabottom',
    duration=10,
    size=(200,200),
    bbox=dict(width=300, height=300),
)

magnitudes = range(1, 6)
reward_duration = 1  # in seconds
interpulse_interval = 0.2  # in seconds
config['magnitude_mapping'] = {
    mag: {
        'n_pulses': mag, 
        'duration': reward_duration, 
        'interpulse_interval': interpulse_interval
    } 
    for mag in magnitudes
}
images = [f'stimuli/aada{chr(97+i)}.png' for i in range(5)]
config['items'] = dict(zip(magnitudes, images))

CENTER = 1080//2, 1920//2
OFFSET = 250
LEFT = (CENTER[0] - OFFSET, CENTER[1])
RIGHT = (CENTER[0] + OFFSET, CENTER[1])
config['locations'] = {
    'left': LEFT,
    'right': RIGHT,
    'center': CENTER,
}

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
#         length=5,
#         retry={'timeout': True},
#     )
# # then one block mixing all forced choice trials
# blocks[len(blocks)] = dict(
#     conditions=[f'f{mag}left' for mag in magnitudes] + [f'f{mag}right' for mag in magnitudes],
#     length=50,
#     retry={'timeout': True},
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
    previous_block = f'valuediff({min(value_difference+1, 4)})'
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