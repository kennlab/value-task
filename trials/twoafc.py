from experiment.trial import Trial, TrialResult
from experiment.experiments.adapters import ImageAdapter, TouchAdapter, RewardAdapter, TimeCounter, RectAdapter
from experiment.experiments.scene import Scene
from experiment.util.bbox import T_BBOX_SPEC
from typing import Any, Optional, Tuple

INTERPULSE_INTERVAL = 0.2
REWARD_BAR_WIDTH = 0.0833333333
REWARD_BAR_HEIGHT_PER_LEVEL = 0.1388888889
REWARD_PROGRESS_SIZE = (0.4166666667, 0.0925925926)
REWARD_PROGRESS_GAP = 0.0185185185
HIDDEN_PROGRESS_SIZE = (0.001, 0.001)


class TwoAFCTrial(Trial):
    CHOICE_NAMES = ('option1', 'option2')
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
    CENTER = (0.0, 0.0)

    def __init__(
        self,
        options: Tuple[str, str],
        magnitudes: Tuple[int, int],
        locs: Tuple[Tuple[float, float], Tuple[float, float]],
        magnitude_mapping=None,
        duration: float = 5.0,
        size: Tuple[float, float] = (200, 200),
        bbox: Optional[T_BBOX_SPEC] = None,
        reward_channels: Tuple[int, ...] = (1, 2),
        center=CENTER,
        cue_incorrect: bool = False,
        reward_feedback_method: str = 'bar_height',
        coordinate_space: str = 'ndc',
        stimulus_set: int | None = None,
    ):
        super().__init__()
        self.options = options
        self.magnitudes = magnitudes
        self.locs = locs
        self.magnitude_mapping = magnitude_mapping or self.DEFAULT_MAGNITUDE_MAPPING
        self.size = size
        self.duration = duration
        if bbox is None:
            self.bbox: T_BBOX_SPEC = {'width': size[0] * 1.5, 'height': size[1] * 1.5}
        else:
            self.bbox = bbox
        self.reward_channels = reward_channels
        self.error_duration = 2.0
        self.timeout_duration = 2.0
        self.center = center
        self.cue_incorrect = cue_incorrect
        self.reward_feedback_method = reward_feedback_method
        self.coordinate_space = coordinate_space
        self.stimulus_set = stimulus_set

    @classmethod
    def from_config(cls, config: dict) -> 'TwoAFCTrial':
        magnitudes = tuple(config['magnitudes'])
        options = tuple(config['items'][mag] for mag in magnitudes)
        locs = tuple(config['locations'][loc] for loc in config['locs'])
        magnitude_mapping = config.get('magnitude_mapping', cls.DEFAULT_MAGNITUDE_MAPPING)
        duration = config.get('duration', 5.0)
        size = tuple(config.get('size', (200, 200)))
        bbox = config.get('bbox', None)
        reward_channels = tuple(config.get('reward_channels', (1, 2)))
        center = tuple(config['locations'].get('center', cls.CENTER))
        cue_incorrect = config.get('cue_incorrect', False)
        reward_feedback_method = config.get('reward_feedback_method', 'bar_height')
        coordinate_space = config.get('coordinate_space', 'ndc')
        stimulus_set = config.get('stimulus_set')
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
            cue_incorrect=cue_incorrect,
            reward_feedback_method=reward_feedback_method,
            coordinate_space=coordinate_space,
            stimulus_set=stimulus_set,
        )

    def reward_params_for_choices(self) -> list[dict[str, Any]]:
        return [self.magnitude_mapping[mag] for mag in self.magnitudes]

    def trial_data(self, reward_params) -> dict[str, Any]:
        return {
            "trial_kind": "magnitude_choice",
            "options": self.options,
            "magnitudes": self.magnitudes,
            "locations": self.locs,
            "reward_params": reward_params,
            "stimulus_set": self.stimulus_set,
        }

    def build_targets(self) -> dict[str, ImageAdapter]:
        return {
            choice_name: ImageAdapter(
                image=self.options[index],
                position=self.locs[index],
                size=self.size,
                coordinate_space=self.coordinate_space,
                bbox=self.bbox,
            )
            for index, choice_name in enumerate(self.CHOICE_NAMES)
        }

    def get_choice_scene(self, mgr) -> tuple[Scene, TouchAdapter]:
        tc = TouchAdapter(
            time_counter=self.duration,
            items=self.build_targets(),
            allow_outside_touch=True,
        )
        return Scene(mgr, adapter=tc), tc

    def correct_choice(self) -> str:
        return 'option1' if self.magnitudes[0] > self.magnitudes[1] else 'option2'

    def result_for_choice(self, chosen: str, data: dict[str, Any], reward_params) -> TrialResult:
        outcome = 'correct' if chosen == self.correct_choice() else 'incorrect'
        data['chosen'] = chosen
        return TrialResult(
            continue_session=True,
            outcome=outcome,
            data=data,
        )

    def timeout_result(self, data: dict[str, Any]) -> TrialResult:
        return TrialResult(
            continue_session=True,
            outcome="timeout",
            data=data,
        )

    def reward_adapter_kwargs(self, magnitude_level: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.reward_feedback_method == 'bar_height':
            kwargs['children'] = [
                RectAdapter(
                    position=self.center,
                    size=(REWARD_BAR_WIDTH, REWARD_BAR_HEIGHT_PER_LEVEL * magnitude_level),
                    colour='#000000',
                    coordinate_space=self.coordinate_space,
                )
            ]
            kwargs['progress_params'] = dict(
                position=self.center,
                size=HIDDEN_PROGRESS_SIZE,
                colour=(0, 0, 0),
                gap=0.0,
                coordinate_space=self.coordinate_space,
            )
        elif self.reward_feedback_method == 'progress':
            kwargs['progress_params'] = dict(
                position=self.center,
                size=REWARD_PROGRESS_SIZE,
                colour=(0, 0, 0),
                gap=REWARD_PROGRESS_GAP,
                coordinate_space=self.coordinate_space,
            )
        return kwargs

    def get_reward_scene(self, mgr, reward_params, magnitude_level, background) -> Scene:
        rew = RewardAdapter.from_manager(
            manager=mgr,
            channels=self.reward_channels,
            **reward_params,
            **self.reward_adapter_kwargs(magnitude_level),
        )
        return Scene(mgr, rew, background=background)

    def outcome_scene_for_result(
        self,
        mgr,
        result: TrialResult,
        chosen: str | None,
        data: dict[str, Any],
        reward_params,
    ) -> Scene:
        if result.outcome == 'timeout':
            return Scene(
                mgr,
                adapter=TimeCounter(self.timeout_duration),
                background=self.backgrounds['timeout'],
            )

        chosen_index = self.CHOICE_NAMES.index(chosen)
        chosen_reward = reward_params[chosen_index]
        chosen_mag_level = self.magnitudes[chosen_index]
        background = self.backgrounds['correct']
        if result.outcome == 'incorrect' and self.cue_incorrect:
            background = self.backgrounds['incorrect']
        return self.get_reward_scene(
            mgr,
            chosen_reward,
            chosen_mag_level,
            background=background,
        )

    def run(self, mgr) -> TrialResult:
        reward_params = self.reward_params_for_choices()
        data = self.trial_data(reward_params)
        scene, tc = self.get_choice_scene(mgr)

        scene.run()
        data['RT'] = tc.RT
        if scene.quit:
            return TrialResult(
                continue_session=False,
                outcome="quit",
                data=data,
            )

        if tc.chosen in self.CHOICE_NAMES:
            res = self.result_for_choice(tc.chosen, data, reward_params)
        else:
            res = self.timeout_result(data)

        outcome_scene = self.outcome_scene_for_result(mgr, res, tc.chosen, data, reward_params)
        outcome_scene.run()
        if outcome_scene.quit:
            res.continue_session = False

        print(data)
        mgr.record(**data, outcome=res.outcome)
        return res
