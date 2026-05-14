import random
from typing import Any, Mapping, Optional, Tuple

from experiment.experiments.adapters import ImageAdapter, RewardAdapter, TimeCounter
from experiment.experiments.scene import Scene
from experiment.trial import TrialResult
from experiment.util.bbox import T_BBOX_SPEC

from trials.twoafc import HIDDEN_PROGRESS_SIZE, TwoAFCTrial


class DistributionTwoAFCTrial(TwoAFCTrial):
    def __init__(
        self,
        distribution_options: Tuple[str, str],
        distribution_cues: Mapping[str, Mapping[str, Any]],
        magnitude_items: Mapping[int, str],
        locs: Tuple[Tuple[float, float], Tuple[float, float]],
        magnitude_mapping=None,
        duration: float = 5.0,
        size: Tuple[float, float] = (200, 200),
        bbox: Optional[T_BBOX_SPEC] = None,
        reward_channels: Tuple[int, ...] = (1, 2),
        center=TwoAFCTrial.CENTER,
        coordinate_space: str = 'ndc',
        stimulus_set: int | None = None,
    ):
        self.distribution_options = tuple(distribution_options)
        self.distribution_cues = {
            str(cue_id): dict(cue)
            for cue_id, cue in distribution_cues.items()
        }
        self.magnitude_items = {
            int(magnitude): image
            for magnitude, image in magnitude_items.items()
        }
        self._validate_distribution_options()
        cue_images = tuple(
            self.distribution_cues[cue_id]['image']
            for cue_id in self.distribution_options
        )
        super().__init__(
            options=cue_images,
            magnitudes=(0, 0),
            locs=locs,
            magnitude_mapping=magnitude_mapping,
            duration=duration,
            size=size,
            bbox=bbox,
            reward_channels=reward_channels,
            center=center,
            cue_incorrect=False,
            reward_feedback_method='sampled_image',
            coordinate_space=coordinate_space,
            stimulus_set=stimulus_set,
        )

    @classmethod
    def from_config(cls, config: dict) -> 'DistributionTwoAFCTrial':
        distribution_options = tuple(str(cue_id) for cue_id in config['distribution_options'])
        distribution_cues = config['distribution_cues']
        stimulus_set = config.get('stimulus_set')
        magnitude_items = config.get('items')
        if magnitude_items is None:
            magnitude_items = config['stimulus_sets'][stimulus_set]
        locs = tuple(config['locations'][loc] for loc in config['locs'])
        magnitude_mapping = config.get('magnitude_mapping', cls.DEFAULT_MAGNITUDE_MAPPING)
        duration = config.get('duration', 5.0)
        size = tuple(config.get('size', (200, 200)))
        bbox = config.get('bbox', None)
        reward_channels = tuple(config.get('reward_channels', (1, 2)))
        center = tuple(config['locations'].get('center', cls.CENTER))
        coordinate_space = config.get('coordinate_space', 'ndc')
        return cls(
            distribution_options=distribution_options,
            distribution_cues=distribution_cues,
            magnitude_items=magnitude_items,
            locs=locs,
            magnitude_mapping=magnitude_mapping,
            duration=duration,
            size=size,
            bbox=bbox,
            reward_channels=reward_channels,
            center=center,
            coordinate_space=coordinate_space,
            stimulus_set=stimulus_set,
        )

    def _validate_distribution_options(self) -> None:
        for cue_id in self.distribution_options:
            if cue_id not in self.distribution_cues:
                raise ValueError(f"Unknown distribution cue: {cue_id}")
            cue = self.distribution_cues[cue_id]
            values = cue.get('magnitude_values', [])
            probabilities = cue.get('probabilities', [])
            if len(values) != len(probabilities):
                raise ValueError(f"Distribution cue {cue_id} has mismatched values/probabilities.")
            if not values:
                raise ValueError(f"Distribution cue {cue_id} has no magnitude values.")
            if sum(float(probability) for probability in probabilities) <= 0:
                raise ValueError(f"Distribution cue {cue_id} probabilities must sum above zero.")
            missing_images = set(int(value) for value in values) - set(self.magnitude_items)
            if missing_images:
                raise ValueError(
                    f"Stimulus set {self.stimulus_set} lacks magnitude images for cue "
                    f"{cue_id}: {sorted(missing_images)}"
                )

    def reward_params_for_choices(self):
        return None

    def trial_data(self, reward_params) -> dict[str, Any]:
        return {
            "trial_kind": "distribution_choice",
            "distribution_options": self.distribution_options,
            "distribution_images": self.options,
            "distribution_probabilities": {
                cue_id: list(self.distribution_cues[cue_id]['probabilities'])
                for cue_id in self.distribution_options
            },
            "distribution_magnitude_values": {
                cue_id: list(self.distribution_cues[cue_id]['magnitude_values'])
                for cue_id in self.distribution_options
            },
            "locations": self.locs,
            "stimulus_set": self.stimulus_set,
        }

    def sample_magnitude(self, cue_id: str) -> int:
        cue = self.distribution_cues[cue_id]
        magnitude_values = [int(value) for value in cue['magnitude_values']]
        probabilities = [float(probability) for probability in cue['probabilities']]
        return random.choices(magnitude_values, weights=probabilities, k=1)[0]

    def result_for_choice(self, chosen: str, data: dict[str, Any], reward_params) -> TrialResult:
        chosen_index = self.CHOICE_NAMES.index(chosen)
        chosen_distribution = self.distribution_options[chosen_index]
        sampled_magnitude = self.sample_magnitude(chosen_distribution)
        sampled_reward_params = self.magnitude_mapping[sampled_magnitude]
        sampled_image = self.magnitude_items[sampled_magnitude]

        data.update({
            "chosen": chosen,
            "chosen_distribution": chosen_distribution,
            "chosen_distribution_image": self.options[chosen_index],
            "sampled_magnitude": sampled_magnitude,
            "sampled_magnitude_image": sampled_image,
            "sampled_reward_params": sampled_reward_params,
        })
        return TrialResult(
            continue_session=True,
            outcome="choice",
            data=data,
        )

    def get_reward_scene(self, mgr, reward_params, magnitude_level, background) -> Scene:
        sampled_image = self.magnitude_items[int(magnitude_level)]
        rew = RewardAdapter.from_manager(
            manager=mgr,
            channels=self.reward_channels,
            **reward_params,
            children=[
                ImageAdapter(
                    image=sampled_image,
                    position=self.center,
                    size=self.size,
                    coordinate_space=self.coordinate_space,
                )
            ],
            progress_params=dict(
                position=self.center,
                size=HIDDEN_PROGRESS_SIZE,
                colour=(0, 0, 0),
                gap=0.0,
                coordinate_space=self.coordinate_space,
            ),
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
        return self.get_reward_scene(
            mgr,
            data['sampled_reward_params'],
            data['sampled_magnitude'],
            background=self.backgrounds['correct'],
        )
