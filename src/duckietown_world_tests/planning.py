import os

import numpy as np
from typing import *
import geometry as geo
from comptests import comptest, run_module_tests, get_comptests_output_dir
from duckietown_world import PWMCommands, SampledSequence, draw_static, \
    SE2Transform, DB18, construct_map, iterate_with_dt
from duckietown_world.seqs.tsequence import SampledSequenceBuilder
from duckietown_world.svg_drawing.misc import TimeseriesPlot
from duckietown_world.world_duckietown.pwm_dynamics import get_DB18_nominal
from duckietown_world.world_duckietown.types import TSE2v, se2v
from duckietown_world.world_duckietown.other_objects import Trajectory
from duckietown_world.world_duckietown.utils import get_velocities_from_sequence
from duckietown_world.rules import evaluate_rules, get_scores, RuleEvaluationResult
from duckietown_world.optimization import LexicographicSemiorderTracker, \
    LexicographicTracker, ProductOrderTracker


def get_bundle(N: int, times: List[float]):
    tries = {
    }

    d = 0.1
    b = 0.5

    for i in range(N):
        commands_ssb = SampledSequenceBuilder[PWMCommands]()
        for t in times:
            u_left = np.random.choice([-d + b, b, b + d])
            u_right = np.random.choice([-d + b, b, b + d])
            u = PWMCommands(motor_left=u_left, motor_right=u_right)
            commands_ssb.add(t, u)
        commands = commands_ssb.as_sequence()
        tries[str(i)] = commands
    return tries


def get_trajs_bundle(commands_bundle, q, v):
    parameters = get_DB18_nominal(delay=0)
    trajs_bundle = {}
    for id_try, commands in commands_bundle.items():
        seq = integrate_dynamics2(parameters, q, v, commands)
        trajs_bundle[id_try] = seq
    return trajs_bundle


@comptest
def test_planning():
    # TODO work with delay
    root = get_simple_map()
    parameters = get_DB18_nominal(delay=0)

    # initial configuration
    init_pose = np.array([0, 0.8])
    init_vel = np.array([0, 0])

    q0 = geo.SE2_from_R2(init_pose)
    v0 = geo.se2_from_linear_angular(init_vel, 0)

    q, v = q0, v0

    # number of candidate trajectories
    N = 15
    # planning horizon
    horizon = 4
    # planning abortion
    stop = 12
    steps_per_second = 8

    times = list(np.linspace(0, stop, stop * (steps_per_second + 1)))

    executed_commands_ssb = SampledSequenceBuilder[PWMCommands]()
    # TODO resolve t
    for t in times:
        # get random commands bundle
        # print('Time is : ', t)
        commands_bundle = get_bundle(N, list(np.linspace(t, t + horizon, horizon * (steps_per_second + 1))))
        # get random trajectories
        trajs_bundle = get_trajs_bundle(commands_bundle, q, v)
        # evaluate traj_s bundle, e.g get optimum id
        optimum = evaluate_trajs(commands_bundle, trajs_bundle)
        # get first command
        # TODO resolve id
        opt_items = optimum.popitem()
        opt_id = opt_items[0]
        # TODO resolve hard coding
        # t
        transform = SE2Transform.identity()
        ground_truth_ssb = SampledSequenceBuilder[SE2Transform]()
        ground_truth_ssb.add(t, transform)
        ground_truth = ground_truth_ssb.as_sequence()
        if t != times[-1]:
            for k, traj in trajs_bundle.items():
                if k == opt_id:
                    root.set_object(f'traj{k} at {t}', Trajectory(traj, color="red"), ground_truth=ground_truth)
                else:
                    root.set_object(f'traj{k} at {t}', Trajectory(traj, color="yellow"), ground_truth=ground_truth)

        commands_to_execute = commands_bundle[opt_id]
        # TODO shorten below
        pwm_commands_to_execute = commands_to_execute.values
        u_first = pwm_commands_to_execute[0]

        # execute command
        executed_commands_ssb.add(t, u_first)

        # get new pose and velocity
        pseudo_commands_sbb = SampledSequenceBuilder[PWMCommands]()
        time_stamps = commands_to_execute.timestamps
        # FIXME
        pseudo_commands_sbb.add(time_stamps[0], pwm_commands_to_execute[0])
        pseudo_commands_sbb.add(time_stamps[1], pwm_commands_to_execute[1])
        pseudo_commands_sbb.add(time_stamps[2], pwm_commands_to_execute[2])
        q, v = get_current_pose_and_velocity(pseudo_commands_sbb, q, v)

    executed_commands = executed_commands_ssb.as_sequence()
    executed_traj = integrate_dynamics2(parameters, q0, v0, executed_commands)
    # print(executed_traj)
    # visualize
    outdir = os.path.join(get_comptests_output_dir(), 'together')
    visualize(root, executed_commands, executed_traj, outdir)


def get_current_pose_and_velocity(commands_ssb, q, v):
    parameters = get_DB18_nominal(delay=0)
    commands = commands_ssb.as_sequence()
    traj = integrate_dynamics2(parameters, q, v, commands)
    poses = traj.transform_values(lambda t: t[0])
    velocities = get_velocities_from_sequence(poses).values
    return poses.values[-1], velocities[-1]


def evaluate_trajs(commands_bundle, trajs_bundle):
    # TODO only use trajs_bundle
    # TODO make generic
    rules_list = ['Drivable areas']
    optimal_traj_tracker = LexicographicTracker(rules_list)
    root = get_simple_map()

    for id_try, commands in commands_bundle.items():
        traj = trajs_bundle[id_try]
        ground_truth = traj.transform_values(lambda t: SE2Transform.from_SE2(t[0]))
        root.set_object(id_try, DB18(), ground_truth=ground_truth)
        # poses = traj.transform_values(lambda t: t[0])
        poses_sequence = traj.transform_values(lambda t: SE2Transform.from_SE2(t[0]))
        interval = SampledSequence.from_iterator(enumerate(commands.timestamps))
        evaluated = evaluate_rules(poses_sequence=poses_sequence, interval=interval, world=root, ego_name=id_try)

        scores = get_scores(evaluated)

        optimal_traj_tracker.digest_traj(id_try, scores)

    # TODO get optimum
    optimum = optimal_traj_tracker.get_optimal_trajs()
    return optimum


def visualize(root, commands, traj_executed, outdir):
    timeseries = {}

    ground_truth = traj_executed.transform_values(lambda t: SE2Transform.from_SE2(t[0]))

    ego_name = 'Planned Trajectory'
    root.set_object(ego_name, DB18(), ground_truth=ground_truth)
    poses = traj_executed.transform_values(lambda t: t[0])

    velocities = get_velocities_from_sequence(poses)
    # TODO what's the difference with below
    # v_test = traj.transform_values(lambda t: t[1])

    linear = velocities.transform_values(linear_from_se2)
    angular = velocities.transform_values(angular_from_se2)

    poses_sequence = traj_executed.transform_values(lambda t: SE2Transform.from_SE2(t[0]))
    interval = SampledSequence.from_iterator(enumerate(commands.timestamps))
    evaluated = evaluate_rules(poses_sequence=poses_sequence, interval=interval, world=root, ego_name=ego_name)

    scores = get_scores(evaluated)

    for key, rer in evaluated.items():
        assert isinstance(rer, RuleEvaluationResult)

        for km, evaluated_metric in rer.metrics.items():
            sequences = {}
            sequences[evaluated_metric.title] = evaluated_metric.cumulative
            plots = TimeseriesPlot(f'{ego_name} - {evaluated_metric.title}', evaluated_metric.title, sequences)
            timeseries[f'{ego_name} - {evaluated_metric.title}'] = plots

    # TODO refactor
    sequences = {}
    sequences['motor_left'] = commands.transform_values(lambda _: _.motor_left)
    sequences['motor_right'] = commands.transform_values(lambda _: _.motor_right)
    plots = TimeseriesPlot(f'{ego_name} - PWM commands', 'pwm_commands', sequences)
    timeseries[f'{ego_name} - commands'] = plots

    sequences = {}
    sequences['linear_velocity'] = linear
    sequences['angular_velocity'] = angular
    plots = TimeseriesPlot(f'{ego_name} - Velocities', 'velocities', sequences)
    timeseries[f'{ego_name} - velocities'] = plots

    sequences = {}
    sequences['linear_velocity'] = linear
    sequences['angular_velocity'] = angular
    plots = TimeseriesPlot(f'{ego_name} - Velocities', 'velocities', sequences)
    timeseries[f'{ego_name} - velocities'] = plots

    draw_static(root, outdir, timeseries=timeseries)


def get_simple_map():
    map_data_yaml = """

       tiles:
       - [floor/W,floor/W, floor/W, floor/W, floor/W, floor/W, floor/W, floor/W, floor/W] 
       - [straight/W   , straight/W   , straight/W, straight/W, straight/W, straight/W   , straight/W, straight/W, straight/W]
       - [floor/W,floor/W, floor/W, floor/W, floor/W, floor/W, floor/W, floor/W, floor/W]
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


def integrate_dynamics2(factory, q0, v0, commands: SampledSequence) -> SampledSequence[TSE2v]:
    # starting time
    c0 = q0, v0
    state = factory.initialize(c0=c0, t0=commands.timestamps[0])
    ssb = SampledSequenceBuilder[TSE2v]()
    # ssb.add(t0, state.TSE2_from_state())
    for it in iterate_with_dt(commands):
        ssb.add(it.t0, state.TSE2_from_state())
        state = state.integrate(it.dt, it.v0)

    return ssb.as_sequence()


if __name__ == '__main__':
    run_module_tests()
