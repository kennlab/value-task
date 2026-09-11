from typing import Any, Mapping, Optional, Tuple

from experiment.util.bbox import T_BBOX_SPEC
from experiment.trial import TrialResult
from experiment.experiments.adapters import ImageAdapter, TouchAdapter, RewardAdapter, TimeCounter, RectAdapter
from experiment.experiments.scene import Scene
from trials.twoafc import TwoAFCTrial
from trials.distribution_twoafc import DistributionTwoAFCTrial

from enum import StrEnum, auto

class WinZoneMethod(StrEnum):
    DETERMINISTIC = auto()
    RANDOM = auto()

class WinZoneTrial(DistributionTwoAFCTrial):
    TRIAL_KIND='winzone'
    def __init__(
        self,
        winzone: int,
        winzone_method: WinZoneMethod,
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
        jackpot_pre_delay_duration: float = .2
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
            stimulus_set=stimulus_set
        )
        if 'jackpot' not in self.magnitude_mapping or 'jackpot' not in self.magnitude_items:
            raise ValueError("Stimulus set and mapping must specify a jackpot level")
    @classmethod
    def from_config(cls, config: dict) -> 'WinZoneTrial':
        return cls(
            winzone=config['winzone'],
            winzone_method=WinZoneMethod(config['winzone_method']),
            jackpot_pre_delay_duration=config['jackpot_pre_delay_duration'],
            **super()._config_kwargs(config)
        )
    def trial_data(self) -> dict[str, Any]:
        data = super().trial_data()
        data['winzone'] = self.winzone
        data['winzone_method'] = str(self.winzone_method)
        return data

    def get_winzone_scene(self, mgr) -> tuple[Scene, TouchAdapter]:
        tc = TouchAdapter(
            time_counter=self.duration,
            items={'winzone': ImageAdapter(
                image=self.magnitude_items[self.winzone],
                position=self.CENTER,
                size=self.size,
                coordinate_space=self.coordinate_space,
                bbox=self.bbox
            )},
            allow_outside_touch=True,
        )
        return Scene(mgr, adapter=tc), tc
    
    def run(self, mgr) -> TrialResult:
        data = self.trial_data()
        winzone_scene, winzone_tc = self.get_winzone_scene(mgr)
        scene, tc = self.get_choice_scene(mgr)
        jackpot_pre_delay = Scene(mgr, TimeCounter(self.jackpot_pre_delay_duration))
        jackpot = self.get_reward_scene(
            mgr,
            self.magnitude_mapping["jackpot"],
            "jackpot",
            background=self.backgrounds['correct'],
        )

        winzone_scene.run()
        data['winzone_RT'] = winzone_tc.RT
        if winzone_scene.quit:
            return TrialResult(
                continue_session=False,
                outcome="quit",
                data=data,
            )
        elif winzone_tc.chosen != 'winzone':
            res = TrialResult(continue_session=True, outcome='timeout', data=data)
            outcome_scene = self.outcome_scene_for_result(mgr, res, tc.chosen, data)
            outcome_scene.run()
            if outcome_scene.quit:
                res.continue_session = False

            mgr.record(**data, outcome=res.outcome)
            return res

        scene.run()
        data['RT'] = tc.RT
        if scene.quit:
            return TrialResult(
                continue_session=False,
                outcome="quit",
                data=data,
            )

        if self.winzone_method == WinZoneMethod.RANDOM:
            if tc.chosen in self.CHOICE_NAMES:
                res = self.result_for_choice(tc.chosen, data)
            else:
                res = self.timeout_result(data)

            outcome_scene = self.outcome_scene_for_result(mgr, res, tc.chosen, data)
            outcome_scene.run()
            mgr.record(**data, outcome=res.outcome)
            if outcome_scene.quit:
                res.continue_session = False
                return res
            deliver_jackpot = data.get('sampled_magnitude') == self.winzone
            mgr.record(jackpot_delivered=deliver_jackpot)
            if deliver_jackpot:
                jackpot_pre_delay.run()
                if jackpot_pre_delay.quit:
                    res.continue_session = False
                    return res
                jackpot.run()
                if jackpot.quit:
                    res.continue_session = False
            return res
        elif self.winzone_method == WinZoneMethod.DETERMINISTIC:
            #TODO: check if the rationally optimal answer was chosen and only reinforce then
            pass

            mgr.record(**data, outcome=res.outcome)
            return res
