from enum import StrEnum, auto
from typing import Any, Mapping, Optional, Sequence, Tuple

from experiment.experiments.adapters import BaseAdapter, ImageAdapter, TimeCounter, TouchAdapter
from experiment.experiments.scene import Scene
from experiment.trial import TrialResult
from experiment.util.bbox import T_BBOX_SPEC

from trials.distribution_twoafc import DistributionTwoAFCTrial
from trials.twoafc import TwoAFCTrial


class WinZoneMethod(StrEnum):
    DETERMINISTIC = auto()
    RANDOM = auto()


class WinZoneTrial(DistributionTwoAFCTrial):
    TRIAL_KIND = 'winzone'
    DEFAULT_BACKGROUND = (48, 40, 8)
    REWARD_BACKGROUNDS = {
        'non_jackpot': (0, 255, 0),
        'jackpot': (255, 191, 0),
    }

    def __init__(
        self,
        winzone: int,
        winzone_method: WinZoneMethod,
        winzone_loc: Tuple[float, float],
        distribution_options: Tuple[str, ...],
        distribution_cues: Mapping[str, Mapping[str, Any]],
        magnitude_items: Mapping[int, str],
        locs: Tuple[Tuple[float, float], ...],
        magnitude_mapping=None,
        duration: float = 5.0,
        size: Tuple[float, float] = (200, 200),
        bbox: Optional[T_BBOX_SPEC] = None,
        reward_channels: Tuple[int, ...] = (1, 2),
        center=TwoAFCTrial.CENTER,
        coordinate_space: str = 'ndc',
        stimulus_set: int | None = None,
        jackpot_pre_delay_duration: float = .2,
        persist_winzone_cue: bool = False,
        persistent_winzone_position: Optional[Tuple[float, float]] = None,
        persistent_winzone_size: Optional[Tuple[float, float]] = None,
    ):
        self.winzone = winzone
        self.winzone_method = winzone_method
        if winzone_method == WinZoneMethod.DETERMINISTIC:
            raise NotImplementedError("Only WinZoneMethod random is currently implemented")
        self.jackpot_pre_delay_duration = jackpot_pre_delay_duration
        super().__init__(
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
        self.winzone_loc = winzone_loc
        self.persist_winzone_cue = persist_winzone_cue
        self.persistent_winzone_position = (
            tuple(persistent_winzone_position)
            if persistent_winzone_position is not None
            else winzone_loc
        )
        self.persistent_winzone_size = (
            tuple(persistent_winzone_size)
            if persistent_winzone_size is not None
            else self.size
        )
        if 'jackpot' not in self.magnitude_mapping or 'jackpot' not in self.magnitude_items:
            raise ValueError("Stimulus set and mapping must specify a jackpot level")

    @classmethod
    def from_config(cls, config: dict) -> 'WinZoneTrial':
        return cls(
            winzone=config['winzone'],
            winzone_method=WinZoneMethod(config['winzone_method']),
            winzone_loc=config['winzone_loc'],
            jackpot_pre_delay_duration=config.get('jackpot_pre_delay_duration', .2),
            persist_winzone_cue=config.get('persist_winzone_cue', False),
            persistent_winzone_position=config.get('persistent_winzone_position'),
            persistent_winzone_size=config.get('persistent_winzone_size'),
            **super()._config_kwargs(config),
        )

    def trial_data(self) -> dict[str, Any]:
        data = super().trial_data()
        data['winzone'] = self.winzone
        data['winzone_method'] = str(self.winzone_method)
        data['persist_winzone_cue'] = self.persist_winzone_cue
        data['persistent_winzone_position'] = self.persistent_winzone_position
        data['persistent_winzone_size'] = self.persistent_winzone_size
        return data

    def _winzone_image(self, position, size) -> ImageAdapter:
        return ImageAdapter(
            image=self.magnitude_items[self.winzone],
            position=position,
            size=size,
            coordinate_space=self.coordinate_space,
            bbox=self.bbox,
        )

    def _persistent_cue_adapters(self) -> list[BaseAdapter]:
        if not self.persist_winzone_cue:
            return []
        return [
            self._winzone_image(
                self.persistent_winzone_position,
                self.persistent_winzone_size,
            ),
        ]

    def get_winzone_scene(self, mgr) -> tuple[Scene, TouchAdapter]:
        tc = TouchAdapter(
            time_counter=self.duration,
            items={'winzone': self._winzone_image(self.winzone_loc, self.size)},
            allow_outside_touch=True,
        )
        return Scene(mgr, adapter=tc, background=self.DEFAULT_BACKGROUND), tc

    def outcome_scene_for_result(
        self,
        mgr,
        result: TrialResult,
        chosen: str | None,
        data: dict[str, Any],
        aux_adapters: Optional[Sequence[BaseAdapter]] = None,
    ) -> Scene:
        if result.outcome == 'timeout':
            return super().outcome_scene_for_result(
                mgr,
                result,
                chosen,
                data,
                aux_adapters=aux_adapters,
            )
        background_name = 'jackpot' if data['jackpot_delivered'] else 'non_jackpot'
        return self.get_reward_scene(
            mgr,
            data['sampled_reward_params'],
            data['sampled_magnitude'],
            background=self.REWARD_BACKGROUNDS[background_name],
            aux_adapters=aux_adapters,
        )

    @staticmethod
    def _quit_result(data: dict[str, Any]) -> TrialResult:
        return TrialResult(continue_session=False, outcome='quit', data=data)

    def _run_and_record_outcome(
        self,
        mgr,
        result: TrialResult,
        chosen: str | None,
        data: dict[str, Any],
    ) -> None:
        outcome_scene = self.outcome_scene_for_result(
            mgr,
            result,
            chosen,
            data,
            aux_adapters=self._persistent_cue_adapters(),
        )
        outcome_scene.run()
        if outcome_scene.quit:
            result.continue_session = False
        mgr.record(**data, outcome=result.outcome)

    def _run_jackpot(self, mgr, result: TrialResult) -> None:
        pre_delay = Scene(
            mgr,
            TimeCounter(self.jackpot_pre_delay_duration),
            aux_adapters=self._persistent_cue_adapters(),
            background=self.DEFAULT_BACKGROUND,
        )
        pre_delay.run()
        if pre_delay.quit:
            result.continue_session = False
            return

        jackpot = self.get_reward_scene(
            mgr,
            self.magnitude_mapping['jackpot'],
            'jackpot',
            background=self.REWARD_BACKGROUNDS['jackpot'],
            aux_adapters=self._persistent_cue_adapters(),
        )
        jackpot.run()
        if jackpot.quit:
            result.continue_session = False

    def run(self, mgr) -> TrialResult:
        data = self.trial_data()
        winzone_scene, winzone_tc = self.get_winzone_scene(mgr)

        winzone_scene.run()
        data['winzone_RT'] = winzone_tc.RT
        if winzone_scene.quit:
            return self._quit_result(data)
        if winzone_tc.chosen != 'winzone':
            data['jackpot_delivered'] = False
            result = self.timeout_result(data)
            self._run_and_record_outcome(mgr, result, None, data)
            return result

        choice_scene, choice_tc = self.get_choice_scene(
            mgr,
            aux_adapters=self._persistent_cue_adapters(),
        )
        choice_scene.run()
        data['RT'] = choice_tc.RT
        if choice_scene.quit:
            return self._quit_result(data)

        if choice_tc.chosen in self.CHOICE_NAMES:
            result = self.result_for_choice(choice_tc.chosen, data)
        else:
            result = self.timeout_result(data)
        data['jackpot_delivered'] = data.get('sampled_magnitude') == self.winzone

        self._run_and_record_outcome(mgr, result, choice_tc.chosen, data)
        if not result.continue_session or not data['jackpot_delivered']:
            return result

        self._run_jackpot(mgr, result)
        return result
