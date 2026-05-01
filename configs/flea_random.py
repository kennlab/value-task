from typing import Dict, Any
from itertools import product, combinations
import os

import numpy as np
import pandas as pd

DISPLAY_SIZE = (1920, 1080)
ASPECT_RATIO = DISPLAY_SIZE[0] / DISPLAY_SIZE[1]
STIMULUS_WIDTH = 0.3
STIMULUS_HEIGHT = STIMULUS_WIDTH * ASPECT_RATIO
STIMULUS_SIZE = (STIMULUS_WIDTH, STIMULUS_HEIGHT)
BBOX_RATIO = 1.5
STIMULUS_BBOX = {'width': STIMULUS_WIDTH * BBOX_RATIO, 'height': STIMULUS_HEIGHT * BBOX_RATIO}
N_POSITIONS = 6
RADIUS = 0.5
ANGLES = np.arange(N_POSITIONS) * (360/N_POSITIONS) * np.pi/180
x, y = RADIUS * np.cos(ANGLES), RADIUS * np.sin(ANGLES)
STIMULUS_LOCATIONS = list(zip(x, y))
LOCATIONS: Dict[int|str, tuple] = {
    'center': (0.0, 0.0),
}
for i, (x, y) in enumerate(STIMULUS_LOCATIONS):
    LOCATIONS[i] = (x, y)
locations = range(N_POSITIONS)
location_pairs = [(i, (i+(N_POSITIONS//2))%N_POSITIONS) for i in locations]
MAGNITUDES = tuple(range(1, 6))

# Edit this to choose which stimulus sets can appear in a session.
ENABLED_STIMULUS_SETS = (1, 2, 3, 4, 5)
TRIALS_PER_STIMULUS_SET_BLOCK = 50

config: Dict[str, Any] = dict(
    name='fleabottom',
    coordinate_space='ndc',
    storage={'type': 'sqlite', 'path': 'data/data.db'},
    duration=10,
    size=STIMULUS_SIZE,
    bbox=STIMULUS_BBOX,
    allow_outside_touch=True,
    ITI=1.5,
    cue_incorrect=True,
    reward_feedback_method='bar_height',
    locations=LOCATIONS,
    display={
        'size': DISPLAY_SIZE,
        'display': 0,
        'fullscreen': True
    },
    remote_server={
        'enabled': True,
        'show': False,
        'template_path': 'server',
    }
)
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

magnitudes = MAGNITUDES
reward_duration = 0.4  # in seconds
interpulse_interval = 0.2  # in seconds
config['reward_channels'] = ('1','4')
config['magnitude_mapping'] = {
    mag: {'duration': reward_duration*mag, 'n_pulses': 1, 'interpulse_interval': 0}
    for mag in magnitudes
}

image_data = pd.read_csv('stimuli/stimuli.csv')
STIMULUS_SETS = {
    int(stimulus_set_id): {
        int(row.magnitude): row.image
        for row in group.itertuples(index=False)
    }
    for stimulus_set_id, group in image_data.groupby('stimulus_set_id')
}
if not ENABLED_STIMULUS_SETS:
    raise ValueError("ENABLED_STIMULUS_SETS must contain at least one stimulus set.")
unknown_sets = set(ENABLED_STIMULUS_SETS) - set(STIMULUS_SETS)
if unknown_sets:
    raise ValueError(f"Unknown stimulus sets requested: {sorted(unknown_sets)}")
missing_magnitudes = {
    set_id: sorted(set(magnitudes) - set(STIMULUS_SETS[set_id]))
    for set_id in ENABLED_STIMULUS_SETS
    if set(magnitudes) - set(STIMULUS_SETS[set_id])
}
if missing_magnitudes:
    raise ValueError(f"Stimulus sets are missing magnitudes: {missing_magnitudes}")

config['stimulus_sets'] = STIMULUS_SETS
# Fallback only; each generated condition below overrides items with its own
# stimulus set so trials never mix images across sets.
config['items'] = STIMULUS_SETS[ENABLED_STIMULUS_SETS[0]]

forced_choice_trials = {
    f'set{set_id}_forced_m{mag}_loc{loc}': dict(
        stimulus_set=set_id,
        items=STIMULUS_SETS[set_id],
        magnitude=mag,
        loc=loc,
        trial_type='forced'
    )
    for set_id in ENABLED_STIMULUS_SETS
    for mag, loc in product(magnitudes, locations)
}

two_afc_trials = {
    f'set{set_id}_choice_m{option1}v{option2}_loc{locs[0]}v{locs[1]}': dict(
        stimulus_set=set_id,
        items=STIMULUS_SETS[set_id],
        magnitudes=(option1, option2),
        locs=locs,
        trial_type='choice'
    )
    for set_id in ENABLED_STIMULUS_SETS
    for (option1, option2), locs in product(combinations(magnitudes, 2), location_pairs)
}

config['conditions'] = {**forced_choice_trials, **two_afc_trials}

# Random 50-trial blocks by stimulus set. Each block includes all magnitude
# combinations and all opposite-position pairings for that stimulus set.
stimulus_set_block_names = {
    set_id: f'stimulus_set_{set_id}'
    for set_id in ENABLED_STIMULUS_SETS
}
blocks = {}
for set_id in ENABLED_STIMULUS_SETS:
    current_block = stimulus_set_block_names[set_id]
    next_from = list(set(stimulus_set_block_names.values()) - {current_block})
    blocks[current_block] = dict(
        stimulus_set=set_id,
        conditions=[
            condition_name
            for condition_name, condition in two_afc_trials.items()
            if condition['stimulus_set'] == set_id
        ],
        length=TRIALS_PER_STIMULUS_SET_BLOCK,
        retry={'timeout': True},
        transition=[{'next_from': next_from}],
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
config['valid_times'] = [
    {'start': '08:00', 'end': '18:00'}
]
if __name__ == '__main__':
    import pprint
    pprint.pprint(config)
