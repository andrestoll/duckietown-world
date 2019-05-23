import numpy as np
from typing import *
from duckietown_world.world_duckietown import PWMCommands
from duckietown_world.seqs import SampledSequence, iterate_with_dt
from duckietown_world.seqs.tsequence import SampledSequenceBuilder
from duckietown_world.world_duckietown.platform_dynamics import PlatformDynamicsFactory
from duckietown_world.world_duckietown.types import TSE2v

__all__ = [
    'get_random_trajs_bundle',
    'get_random_commands_bundle',
    'integrate_dynamics2',
]


def get_random_commands_bundle(n: int, times: List[float]) -> Dict[str, SampledSequence[PWMCommands]]:
    """
    Creates n random command sequences.
    :param n: Number of different command sequences
    :param times: Duration of command sequence
    :return: Bundle of command sequences
    """
    tries = {
    }

    d = 0.1
    b = 0.5
    # TODO remove after testing
    r = np.random.RandomState(1234)
    for i in range(n):
        commands_ssb = SampledSequenceBuilder[PWMCommands]()
        for t in times:
            u_left = r.choice([-d + b, b, b + d])
            u_right = r.choice([-d + b, b, b + d])
            u = PWMCommands(motor_left=u_left, motor_right=u_right)
            commands_ssb.add(t, u)
        commands = commands_ssb.as_sequence()
        tries[str(i)] = commands
    return tries


def get_random_trajs_bundle(factory: PlatformDynamicsFactory, n: int, times: List[float], q0, v0):
    """
    Creates n random trajectories.
    :param factory: vehicle parameters
    :param n: Number of trajectories
    :param times: Duration of command sequence
    :param q0: Initial pose
    :param v0: Initial velocity
    :return: Trajectory bundle
    """
    commands_bundle = get_random_commands_bundle(n, times)
    trajs_bundle = {}
    for id_try, commands in commands_bundle.items():
        seq = integrate_dynamics2(factory, q0, v0, commands)
        trajs_bundle[id_try] = seq
        # TODO why SampledSequence[PWMCommands]
    return trajs_bundle, commands_bundle


def integrate_dynamics2(factory: PlatformDynamicsFactory, q0, v0, commands: SampledSequence) -> SampledSequence[TSE2v]:
    """
    Integrates the dynamics for given vehicle and PWM command sequence.
    :param factory: Vehicle parameters
    :param q0: Initial pose
    :param v0: Initial velocity
    :param commands: PWM command sequence
    :return: Sampled sequence of poses and velocities
    """
    # starting time
    c0 = q0, v0
    state = factory.initialize(c0=c0, t0=commands.timestamps[0])
    ssb = SampledSequenceBuilder[TSE2v]()
    # ssb.add(t0, state.TSE2_from_state())
    for it in iterate_with_dt(commands):
        ssb.add(it.t0, state.TSE2_from_state())
        state = state.integrate(it.dt, it.v0)

    a = ssb.as_sequence()

    return ssb.as_sequence()
