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
reward_durations = np.arange(5) * 0.5 + 1.5
config['magnitude_mapping'] = dict(zip(magnitudes, reward_durations))
images = [f'stimuli/aada{chr(97+i)}.png' for i in range(5)]
config['items'] = dict(zip(magnitudes, images))

CENTER = (640, 360)
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
for i, mag in enumerate(magnitudes):
    blocks[i] = dict(
        conditions=[f'f{mag}left', f'f{mag}right'],
        length=10,
        retry={'timeout': True},
    )
## then we will have blocks of 2AFC trials mixing all magnitude levels, starting with easy trials
for value_difference in range(4, 0, -1):
    condition_list = []
    for option1 in magnitudes:
        option2 = option1 - value_difference
        if option2 in magnitudes:
            condition_list.append(f'c{option1}v{option2}_opt1left')
            condition_list.append(f'c{option1}v{option2}_opt1right')

    block_id = len(blocks)
    blocks[block_id] = dict(
        conditions=condition_list,
        length=20,
        retry={'timeout': True},
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