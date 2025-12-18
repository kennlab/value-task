from experiment.trial import Trial, TrialResult
from experiment.experiments.adapters import ImageAdapter, TouchAdapter, RewardAdapter, TimeCounter
from experiment.experiments.scene import Scene
from experiment.util.bbox import T_BBOX_SPEC
from typing import Optional, Tuple

INTERPULSE_INTERVAL = 0.2
class ForcedChoiceTrial(Trial):
    DEFAULT_MAGNITUDE_MAPPING = {
        1: 1,
        2: 1.5,
        3: 2,
        4: 2.5,
        5: 3,
    }
    backgrounds = {
        'correct': (0, 255, 0),
        'incorrect': (255, 0, 0),
        'timeout': (0, 0, 255),
    }
    CENTER = (640, 360)
    def __init__(self, 
        stimulus: str, 
        magnitude: int, 
        loc: Tuple[int, int],
        magnitude_mapping=None, 
        duration: float=5.0,
        size: Tuple[int, int]=(200, 200), 
        bbox: Optional[T_BBOX_SPEC]=None,
        reward_channels: Tuple[int, ...]=(1, 2),
        center=CENTER
    ):
        super().__init__()
        self.stimulus = stimulus
        self.magnitude = magnitude
        self.loc = loc
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
    
    @classmethod
    def from_config(cls, config: dict) -> 'ForcedChoiceTrial':
        magnitude = config['magnitude']
        stimulus = config['items'][ magnitude ]
        loc = tuple(config['locations'][config['loc']])
        magnitude_mapping = config.get('magnitude_mapping', cls.DEFAULT_MAGNITUDE_MAPPING)
        duration = config.get('duration', 5.0)
        size = tuple(config.get('size', (200, 200)))
        bbox = config.get('bbox', None)
        reward_channels = tuple(config.get('reward_channels', (1, 2)))
        center = tuple(config['locations'].get('center', cls.CENTER))
        return cls(
            stimulus=stimulus,
            magnitude=magnitude,
            loc=loc,
            magnitude_mapping=magnitude_mapping,
            duration=duration,
            size=size,
            bbox=bbox,
            reward_channels=reward_channels,
            center=center
        )

    def run(self, mgr) -> TrialResult:
        reward_duration = self.magnitude_mapping[self.magnitude]
        data = {"stimulus": self.stimulus, "magnitude": self.magnitude, "location": self.loc, "reward_duration": reward_duration}

        target = ImageAdapter(
            image=self.stimulus,
            position=self.loc,
            size=self.size,
            bbox=self.bbox
        )
        tc = TouchAdapter(
            time_counter=self.duration,
            items={'target': target},
        )
        scene = Scene(mgr, adapter=tc)
        rew = RewardAdapter.from_manager(
            manager=mgr,
            duration=reward_duration,
            channels=self.reward_channels,
            n_pulses=1,
            interpulse_interval=INTERPULSE_INTERVAL,
            progress_params=dict(
                position=self.center,
                size=(400, 50),
                colour=(0, 0, 0),
                gap=10
            )
        )
        reward_scene = Scene(mgr, rew, background=self.backgrounds['correct'])
        incorrect_scene = Scene(mgr, adapter=TimeCounter(self.error_duration), background=self.backgrounds['incorrect'])
        timeout_scene = Scene(mgr, adapter=TimeCounter(self.timeout_duration), background=self.backgrounds['timeout'])
        outcome_scenes = {
            'correct': reward_scene,
            'incorrect': incorrect_scene,
            'timeout': timeout_scene
        }

        scene.run()
        if scene.quit:
            return TrialResult(
                continue_session=False, 
                outcome="quit", 
                data=data
            )
        elif tc.chosen == 'target':
            res = TrialResult(
                continue_session=True, 
                outcome="correct", 
                data=data
            )
        else:
            res = TrialResult(
                continue_session=True, 
                outcome="timeout", 
                data=data
            )
        outcome_scene = outcome_scenes[res.outcome]
        outcome_scene.run()
        if outcome_scene.quit:
            res.continue_session = False
        return res