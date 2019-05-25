import os

import numpy as np
from typing import *
import geometry as geo
from comptests import comptest, run_module_tests, get_comptests_output_dir
from duckietown_world import SampledSequence, draw_static, SE2Transform, DB18, DB19, construct_map, Duckie
from duckietown_world.svg_drawing.misc import TimeseriesPlot
from duckietown_world.world_duckietown.pwm_dynamics import get_DB18_nominal
from duckietown_world.world_duckietown.types import se2v
from duckietown_world.utils.trajectory_creation import get_random_trajs_bundle
from duckietown_world.rules import get_scores_of_traj_bundle, evaluate_rules
from duckietown_world.world_duckietown.utils import get_velocities_from_sequence
from duckietown_world.rules import RuleEvaluationResult
from duckietown_world.optimization import LexicographicSemiorderTracker, \
    LexicographicTracker, ProductOrderTracker


@comptest
def test_bundle():
    parameters = get_DB18_nominal(delay=0)

    # initial configuration
    init_pose = np.array([0.5, 0.8])
    init_vel = np.array([0, 0])

    q0 = geo.SE2_from_R2(init_pose)
    v0 = geo.se2_from_linear_angular(init_vel, 0)
    N = 2

    trajs_bundle, commands_bundle = get_random_trajs_bundle(
        parameters, N,
        list(np.linspace(0, 6, 30)),
        q0, v0)

    outdir = os.path.join(get_comptests_output_dir(), 'together')
    visualize(commands_bundle, trajs_bundle, outdir)


def visualize(commands_bundle, trajs_bundle, outdir):
    timeseries = {}
    root = get_simple_map()

    # rules_list = {"Survival time": 0.5, "Drivable areas": 0.5, "Deviation from center line": 0.1}
    rules_list = ["Survival time", "Drivable areas", "Deviation from center line"]

    vehicle = DB18()

    scores_bundle = get_scores_of_traj_bundle(root, commands_bundle, trajs_bundle, vehicle)
    opt_trajs = get_best_trajs(scores_bundle, rules_list)

    for id_try, commands in commands_bundle.items():
        traj = trajs_bundle[id_try]

        ego_name = f'Duckiebot{id_try}'
        ground_truth = traj.transform_values(lambda t: SE2Transform.from_SE2(t[0]))

        if ego_name in opt_trajs.keys():
            root.set_object(ego_name, DB18(name=id_try), ground_truth=ground_truth)
        else:
            root.set_object(ego_name, DB19(name=id_try), ground_truth=ground_truth)

        update_timeseries(timeseries, ego_name, commands, traj)

    draw_static(root, outdir, timeseries=timeseries, scores=scores_bundle)


def update_timeseries(timeseries, ego_name, commands, traj):
    world = get_simple_map()
    poses = traj.transform_values(lambda t: t[0])
    velocities = get_velocities_from_sequence(poses)
    linear = velocities.transform_values(linear_from_se2)
    angular = velocities.transform_values(angular_from_se2)

    ground_truth = traj.transform_values(lambda t: SE2Transform.from_SE2(t[0]))
    world.set_object(ego_name, DB18(), ground_truth=ground_truth)
    poses_sequence = traj.transform_values(lambda t: SE2Transform.from_SE2(t[0]))
    interval = SampledSequence.from_iterator(enumerate(commands.timestamps))

    rules_list = {"Drivable areas": 0.5, "Survival time": 0.5}
    evaluated = evaluate_rules(poses_sequence=poses_sequence, interval=interval, world=world, ego_name=ego_name)

    for key, rer in evaluated.items():
        assert isinstance(rer, RuleEvaluationResult)

        for km, evaluated_metric in rer.metrics.items():
            # if evaluated_metric.title in rules_list.keys():
            sequences = {}
            sequences[f'{evaluated_metric.title} - cumulative'] = evaluated_metric.cumulative
            sequences[f'{evaluated_metric.title} - incremental'] = evaluated_metric.incremental
            plots = TimeseriesPlot(f'{ego_name} - {evaluated_metric.title}', evaluated_metric.title, sequences)
            timeseries[f'{ego_name} - {evaluated_metric.title}'] = plots

    # sequences = {}
    # sequences['motor_left'] = commands.transform_values(lambda _: _.motor_left)
    # sequences['motor_right'] = commands.transform_values(lambda _: _.motor_right)
    # plots = TimeseriesPlot(f'{ego_name} - PWM commands', 'pwm_commands', sequences)
    # timeseries[f'{ego_name} - commands'] = plots
    #
    # sequences = {}
    # sequences['linear_velocity'] = linear
    # sequences['angular_velocity'] = angular
    # plots = TimeseriesPlot(f'{ego_name} - Velocities', 'velocities', sequences)
    # timeseries[f'{ego_name} - velocities'] = plots


def get_best_trajs(scores_bundle, rules_list):
    optimal_traj_tracker = LexicographicTracker(rules_list)

    for ego_name, scores in scores_bundle.items():
        optimal_traj_tracker.digest_traj(ego_name, scores)

    return optimal_traj_tracker.get_optimal_trajs()


def get_simple_map():
    map_data_yaml = """

       tiles:
       - [floor/W,floor/W, floor/W, floor/W, floor/W] 
       - [straight/W   , straight/W   , straight/W, straight/W, straight/W]
       - [floor/W,floor/W, floor/W, floor/W, floor/W]
       tile_size: 0.61
       """

    import yaml
    map_data = yaml.load(map_data_yaml, Loader=yaml.SafeLoader)
    root = construct_map(map_data)
    return root


def linear_from_se2(x: se2v) -> float:
    linear, _ = geo.linear_angular_from_se2(x)
    return linear[0]


def angular_from_se2(x: se2v) -> float:
    _, angular = geo.linear_angular_from_se2(x)
    return angular


if __name__ == '__main__':
    run_module_tests()
