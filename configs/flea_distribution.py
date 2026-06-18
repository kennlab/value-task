from typing import Any, Dict
from itertools import combinations, permutations, product
import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

DISPLAY_SIZE = (1080, 1920)
ASPECT_RATIO = DISPLAY_SIZE[0] / DISPLAY_SIZE[1]
STIMULUS_WIDTH = 0.3
STIMULUS_HEIGHT = STIMULUS_WIDTH * ASPECT_RATIO
STIMULUS_SIZE = (STIMULUS_WIDTH, STIMULUS_HEIGHT)
BBOX_RATIO = 1.5
STIMULUS_BBOX = {'width': STIMULUS_WIDTH * BBOX_RATIO, 'height': STIMULUS_HEIGHT * BBOX_RATIO}
N_POSITIONS = 6
RADIUS = 0.5
ANGLES = np.arange(N_POSITIONS) * (360 / N_POSITIONS) * np.pi / 180
x, y = RADIUS * np.cos(ANGLES), RADIUS * np.sin(ANGLES)
STIMULUS_LOCATIONS = list(zip(x, y))
LOCATIONS: Dict[int | str, tuple] = {
    'center': (0.0, 0.0),
}
for i, (x, y) in enumerate(STIMULUS_LOCATIONS):
    LOCATIONS[i] = (float(x), float(y))

locations = range(N_POSITIONS)
location_pairs = pd.read_csv('configs/locs.csv', index_col=0).loc[0:5, ['x', 'y']].values.tolist()
MAGNITUDES = tuple(range(1, 6))

# Edit this to choose which stimulus sets can appear in a session.
# ENABLED_STIMULUS_SETS = (1, 2, 3, 4, 5)
ENABLED_STIMULUS_SETS = (3,)
MAGNITUDE_TRIALS_PER_STIMULUS_SET_BLOCK = 1
DISTRIBUTION_TRIALS_PER_STIMULUS_SET_BLOCK = 10
DISTRIBUTION_CUE_IDS = ('a', 'b', 'c', 'd')

config: Dict[str, Any] = dict(
    name='fleabottom_distribution',
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
        'display': 1,
        'fullscreen': True,
    },
    remote_server={
        'enabled': True,
        'show': os.getenv('SHOW_REMOTE', 'FALSE').lower() == 'true',
        'template_path': 'server',
    },
)
config['io'] = {
    'reward': {
        'type': 'ISMATEC_SERIAL',
        'address': os.environ.get('PUMP', '/dev/ttyACM0'),
        'channels': [
            {'channel': '1', 'clockwise': True, 'speed': 100},
            {'channel': '4', 'clockwise': True, 'speed': 100},
        ],
    }
}

magnitudes = MAGNITUDES
reward_duration = 0.4
config['reward_channels'] = ('1', '4')
config['magnitude_mapping'] = {
    mag: {'duration': reward_duration * mag, 'n_pulses': 1, 'interpulse_interval': 0}
    for mag in magnitudes
}

with open('stimuli/stimuli.json') as f:
    stimulus_data = json.load(f)
STIMULUS_SETS = {
    int(stimulus_set_id): {}
    for stimulus_set_id in {
        int(cue['stimulus_set_id'])
        for cue in stimulus_data['magnitude_cues']
    }
}
for cue in stimulus_data['magnitude_cues']:
    STIMULUS_SETS[int(cue['stimulus_set_id'])][int(cue['magnitude'])] = cue['image']
DISTRIBUTION_CUES = {
    str(cue['id']): cue
    for cue in stimulus_data.get('distribution_cues', [])
}

if not ENABLED_STIMULUS_SETS:
    raise ValueError("ENABLED_STIMULUS_SETS must contain at least one stimulus set.")
unknown_sets = set(ENABLED_STIMULUS_SETS) - set(STIMULUS_SETS)
if unknown_sets:
    raise ValueError(f"Unknown stimulus sets requested: {sorted(unknown_sets)}")
unknown_distribution_cues = set(DISTRIBUTION_CUE_IDS) - set(DISTRIBUTION_CUES)
if unknown_distribution_cues:
    raise ValueError(f"Unknown distribution cues requested: {sorted(unknown_distribution_cues)}")
missing_magnitudes = {
    set_id: sorted(set(magnitudes) - set(STIMULUS_SETS[set_id]))
    for set_id in ENABLED_STIMULUS_SETS
    if set(magnitudes) - set(STIMULUS_SETS[set_id])
}
if missing_magnitudes:
    raise ValueError(f"Stimulus sets are missing magnitudes: {missing_magnitudes}")

config['stimulus_sets'] = STIMULUS_SETS
config['distribution_cues'] = {
    cue_id: DISTRIBUTION_CUES[cue_id]
    for cue_id in DISTRIBUTION_CUE_IDS
}
config['items'] = STIMULUS_SETS[ENABLED_STIMULUS_SETS[0]]

magnitude_choice_trials = {
    f'set{set_id}_choice_m{option1}v{option2}_loc{locs[0]}v{locs[1]}': dict(
        stimulus_set=set_id,
        items=STIMULUS_SETS[set_id],
        magnitudes=(option1, option2),
        locs=locs,
        trial_type='choice',
    )
    for set_id in ENABLED_STIMULUS_SETS
    for (option1, option2), locs in product(combinations(magnitudes, 2), location_pairs)
}

distribution_choice_trials = {
    f'set{set_id}_dist_{dist1}v{dist2}_loc{locs[0]}v{locs[1]}': dict(
        stimulus_set=set_id,
        items=STIMULUS_SETS[set_id],
        distribution_options=(dist1, dist2),
        locs=locs,
        trial_type='distribution_choice',
    )
    for set_id in ENABLED_STIMULUS_SETS
    for (dist1, dist2), locs in product(permutations(DISTRIBUTION_CUE_IDS, 2), location_pairs)
}

config['conditions'] = {**magnitude_choice_trials, **distribution_choice_trials}

magnitude_block_names = {
    set_id: f'stimulus_set_{set_id}_magnitude'
    for set_id in ENABLED_STIMULUS_SETS
}
distribution_block_names = {
    set_id: f'stimulus_set_{set_id}_distribution'
    for set_id in ENABLED_STIMULUS_SETS
}

blocks = {}
for set_id in ENABLED_STIMULUS_SETS:
    magnitude_block = magnitude_block_names[set_id]
    distribution_block = distribution_block_names[set_id]
    next_magnitude_blocks = [
        block_name
        for next_set_id, block_name in magnitude_block_names.items()
        if next_set_id != set_id
    ] or [magnitude_block]

    blocks[magnitude_block] = dict(
        stimulus_set=set_id,
        conditions=[
            condition_name
            for condition_name, condition in magnitude_choice_trials.items()
            if condition['stimulus_set'] == set_id
        ],
        length=MAGNITUDE_TRIALS_PER_STIMULUS_SET_BLOCK,
        retry={'timeout': True},
        transition=[{'next': distribution_block}],
        method='random',
    )
    blocks[distribution_block] = dict(
        stimulus_set=set_id,
        conditions=[
            condition_name
            for condition_name, condition in distribution_choice_trials.items()
            if condition['stimulus_set'] == set_id
        ],
        length=DISTRIBUTION_TRIALS_PER_STIMULUS_SET_BLOCK,
        retry={'timeout': True},
        transition=[{'next_from': next_magnitude_blocks}],
        method='random',
    )

config['blocks'] = blocks

config['trial_types'] = {
    'choice': {
        'module': 'trials/twoafc.py',
        'class': 'TwoAFCTrial',
    },
    'distribution_choice': {
        'module': 'trials/distribution_twoafc.py',
        'class': 'DistributionTwoAFCTrial',
    },
}
config['hotkeys'] = {
    '4': {'do': 'pump_on'},
    '8': {'do': 'pump_off'},
    '3': {'do': 'pause'},
    '7': {'do': 'unpause'},
    '5': {'do': 'quit'},
}
config['valid_times'] = [
    {'start': '08:00', 'end': '18:00'},
]

if __name__ == '__main__':
    import pprint
    pprint.pprint(config)
