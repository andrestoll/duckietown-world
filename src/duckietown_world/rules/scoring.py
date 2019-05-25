from typing import *
from .rule import evaluate_rules, RuleEvaluationResult
from duckietown_world import SampledSequence, SE2Transform
from ..world_duckietown.other_objects import Vehicle
from ..geo.placed_object import PlacedObject


def get_scores(evaluated: Dict[str, RuleEvaluationResult]) -> Dict[str, float]:
    """
    Returns the scores of a trajectory defined by the RuleEvaluationResult.
    :param evaluated: Dict[name, RuleEvaluationResult]
    :return: scores: Dict[str, float]
    """
    scores = {}
    for key, rer in evaluated.items():
        assert isinstance(rer, RuleEvaluationResult)

        for km, evaluated_metric in rer.metrics.items():
            scores[evaluated_metric.title] = evaluated_metric.total

    return scores


def get_scores_of_traj_bundle(world: PlacedObject,
                              commands_bundle: Dict[str, SampledSequence],
                              trajs_bundle: Dict[str, SampledSequence],
                              vehicle: Vehicle) -> Dict[str, Dict[str, float]]:
    """
    Returns the scores of a trajectory bundle.

    :param world: map for rule evaluation
    :param commands_bundle: bundle of sampled sequence of PWM commands
    :param trajs_bundle: bundle of sampled sequence of posses and velocities
    :param vehicle: Vehicle parameters
    :return: Scores bundle of input trajectory bundle
    """

    scores_of_traj_bundle = {}

    for id_try, commands in commands_bundle.items():
        traj = trajs_bundle[id_try]

        ego_name = f'Duckiebot{id_try}'
        ground_truth = traj.transform_values(lambda t: SE2Transform.from_SE2(t[0]))

        world.set_object(ego_name, vehicle, ground_truth=ground_truth)
        poses_sequence = traj.transform_values(lambda t: SE2Transform.from_SE2(t[0]))
        interval = SampledSequence.from_iterator(enumerate(commands.timestamps))

        evaluated = evaluate_rules(poses_sequence=poses_sequence, interval=interval, world=world, ego_name=ego_name)

        scores = get_scores(evaluated)
        scores_of_traj_bundle[ego_name] = scores

    return scores_of_traj_bundle
