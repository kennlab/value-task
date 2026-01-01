from experiment.trial import Trial, TrialResult
from experiment.experiments.adapters import ImageAdapter, TouchAdapter, RewardAdapter, TimeCounter
from experiment.experiments.scene import Scene
from experiment.util.bbox import T_BBOX_SPEC
from typing import Optional, Tuple

INTERPULSE_INTERVAL = 0.2
class TwoAFCTrial(Trial):
    DEFAULT_MAGNITUDE_MAPPING = {
        1: {'duration': 1},
        2: {'duration': 1.5},
        3: {'duration': 2},
        4: {'duration': 2.5},
        5: {'duration': 3},
    }
    backgrounds = {
        'correct': (0, 255, 0),
        'incorrect': (255, 0, 0),
        'timeout': (0, 0, 255),
    }
    CENTER = (640, 360)
    def __init__(self, 
        options: Tuple[str, str],
        magnitudes: Tuple[int, int],
        locs: Tuple[Tuple[int, int], Tuple[int, int]],
        magnitude_mapping=None, 
        duration: float=5.0,
        size: Tuple[int, int]=(200, 200), 
        bbox: Optional[T_BBOX_SPEC]=None,
        reward_channels: Tuple[int, ...]=(1, 2),
        center=CENTER,
        cue_incorrect: bool = False
    ):
        super().__init__()
        self.options = options
        self.magnitudes = magnitudes
        self.locs = locs
        self.magnitude_mapping = magnitude_mapping or self.DEFAULT_MAGNITUDE_MAPPING
        self.size = size
        self.duration = duration
        if bbox is None:
            self.bbox: T_BBOX_SPEC = {'width': size[0]*1.5, 'height': size[1]*1.5}
        else:
            self.bbox = bbox
        self.reward_channels = reward_channels
        self.error_duration = 2.0
        self.timeout_duration = 2.0
        self.center = center
        self.cue_incorrect = cue_incorrect
    
    @classmethod
    def from_config(cls, config: dict) -> 'TwoAFCTrial':
        magnitudes = config['magnitudes']
        options = tuple( config['items'][ mag ] for mag in magnitudes )
        locs = tuple(config['locations'][loc] for loc in config['locs'])
        magnitude_mapping = config.get('magnitude_mapping', cls.DEFAULT_MAGNITUDE_MAPPING)
        duration = config.get('duration', 5.0)
        size = tuple(config.get('size', (200, 200)))
        bbox = config.get('bbox', None)
        reward_channels = tuple(config.get('reward_channels', (1, 2)))
        center = tuple(config['locations'].get('center', cls.CENTER))
        cue_incorrect = config.get('cue_incorrect', False)
        return cls(
            options=options,
            magnitudes=magnitudes,
            locs=locs,
            magnitude_mapping=magnitude_mapping,
            duration=duration,
            size=size,
            bbox=bbox,
            reward_channels=reward_channels,
            center=center,
            cue_incorrect=cue_incorrect
        )

    def get_reward_scene(self, mgr, reward_params, background) -> Scene:
        rew = RewardAdapter.from_manager(
            manager=mgr, 
            channels=self.reward_channels,
            progress_params=dict(
                position=self.center,
                size=(400, 50),
                colour=(0, 0, 0),
                gap=10
            ),
            **reward_params
        )
        reward_scene = Scene(mgr, rew, background=background)
        return reward_scene

    def run(self, mgr) -> TrialResult:
        reward_params = [self.magnitude_mapping[mag] for mag in self.magnitudes]
        data = {
            "options": self.options, 
            "magnitudes": self.magnitudes, 
            "locations": self.locs, 
            "reward_params": reward_params
        }
        targets = {
            'option1': ImageAdapter(
                image=self.options[0],
                position=self.locs[0],
                size=self.size,
                bbox=self.bbox
            ),
            'option2': ImageAdapter(
                image=self.options[1],
                position=self.locs[1],
                size=self.size,
                bbox=self.bbox
            )
        }
        tc = TouchAdapter(
            time_counter=self.duration,
            items=targets,
        )
        scene = Scene(mgr, adapter=tc)

        if data['magnitudes'][0] > data['magnitudes'][1]:
            correct_choice = 'option1'
        scene.run()
        if scene.quit:
            return TrialResult(
                continue_session=False, 
                outcome="quit", 
                data=data
            )
        elif tc.chosen in ['option1', 'option2']:
            if tc.chosen == correct_choice:
                outcome = 'correct'
            else:
                outcome = 'incorrect'
            data['chosen'] = tc.chosen
            res = TrialResult(
                continue_session=True, 
                outcome=outcome,
                data=data
            )
        else:
            res = TrialResult(
                continue_session=True, 
                outcome="timeout", 
                data=data
            )

        chosen_reward = reward_params[0] if tc.chosen == 'option1' else reward_params[1]
        reward_scene_correct = self.get_reward_scene(
            mgr, 
            chosen_reward, 
            background=self.backgrounds['correct']
        )
        reward_scene_incorrect = self.get_reward_scene(
            mgr, 
            chosen_reward, 
            background=self.backgrounds['incorrect'] if self.cue_incorrect else self.backgrounds['correct']
        )
        timeout_scene = Scene(
            mgr, 
            adapter=TimeCounter(self.timeout_duration), 
            background=self.backgrounds['timeout']
        )
        outcome_scenes = {
            'correct': reward_scene_correct,
            'incorrect': reward_scene_incorrect,
            'timeout': timeout_scene
        }
        outcome_scene = outcome_scenes[res.outcome]
        outcome_scene.run()
        if outcome_scene.quit:
            res.continue_session = False
            
        mgr.record(**data, outcome=res.outcome)
        return res