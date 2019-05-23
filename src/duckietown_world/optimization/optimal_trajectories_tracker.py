from abc import ABCMeta, abstractmethod
from typing import List, Dict
from collections import OrderedDict
import numpy as np

__all__ = [
    'OptimalTrajectoryTracker',
    'assert_valid_rules',
    'strictly_preceedes',
    'LexicographicSemiorderTracker',
    'LexicographicTracker',
    'ProductOrderTracker',
]


class OptimalTrajectoryTracker(metaclass=ABCMeta):
    """
    Keeps track of input trajectories which are optimal
    or might be come optimal in the future (candidate trajectories).

    """

    increasing = ("Survival time", "Drivable areas", "Distance", "Lane distance", "Consecutive lane distance")
    decreasing = ("Deviation from center line", "Deviation from lane direction")
    valid_rules = increasing + decreasing

    @abstractmethod
    def digest_traj(self, egoname: str, scores: Dict[str, float]):
        """
        Digests a new trajectory and evaluates whether it can be discarded or
        whether it has to be kept for further analysis and if other trajectories can be
        discarded upon insertion of new trajectory.

        :param egoname: identifier of trajectory
        :type egoname: str
        :param scores: scores of input trajectory
        :type: scores: Dict[str, float]
        :return: None
        """

    @abstractmethod
    def get_optimal_trajs(self):
        """
        Retrieves identifier and scores of optimal trajectories
        :return: optimal trajectories
        :rtype: Dict[str, Dict[str, float]]
        """


def assert_valid_rules(rules):
    """
    Asserts whether the rules are valid or not
    :param rules: Rules to be asserted
    :return: True if valid otherwise throws ValueError
    """
    for rule in rules:
        if rule not in OptimalTrajectoryTracker.valid_rules:
            raise ValueError(f'{rule} is not valid.')
    return True


def strictly_preceedes(rule: str, score_x: float, score_y: float, slack=0.0):
    """
    Checks whether the score of x for a given rule precedes the score of y.
    Depends on whether scores for a given rule are deemed better f they are higher or lower.

    :param rule: Rule for evaluating x and y
    :param score_x: Score of trajectory x
    :param score_y: Score of trajectory y
    :param slack: Threshold
    :return: True if score of x precedes score of y for given rule
    """
    if rule in OptimalTrajectoryTracker.increasing:
        return score_x - slack > score_y
    else:
        return score_x + slack < score_y


class LexicographicSemiorderTracker(OptimalTrajectoryTracker):
    """
    Optimality is defined in the sense of lexicographic semiordering where
    the optimal trajectories are part of the survivor set where the scores of the rules are
    lexicographically evaluated and all solutions that are within some bound (also called the slack) of
    the absolute optimum are taken to the next step.
    """
    # TODO make generic
    # TODO merge semi and normal lexicographic order
    trajs_tracked: Dict[str, Dict[str, float]]
    rules_with_slack: Dict[str, float]
    title: str

    # TODO change input type of rules, maybe List[Rule]

    def __init__(self, rules_with_slack: Dict[str, float]):
        self.trajs_tracked = dict()
        assert_valid_rules(rules_with_slack.keys())
        self.rules_with_slack = rules_with_slack
        self.title = 'Lexicographic Semiorder'

    def add_collection(self, trajs_collection: Dict[str, Dict[str, float]]):
        self.trajs_tracked = trajs_collection.copy()

    def digest_traj(self, egoname: str, scores: Dict[str, float]):
        if not self.trajs_tracked:
            assert isinstance(scores, Dict)
            self.trajs_tracked[egoname] = scores
            return

        self.__update_trajs(egoname, scores)

    def __update_trajs(self, egoname, scores):
        """
        Upon digestion of a new trajectory, the stored trajectories will be updated.
        If the new trajectory is a candidate for the optimal trajectory set it will be added
        and trajectories which are no longer candidates will be discarded.

        :param egoname: identifier of trajectory
        :type egoname: str
        :param scores: scores of input trajectory
        :type: scores: Dict[str, float]
        :return: None
        """
        for rule in self.rules_with_slack.keys():
            filter_index = []
            slack = self.rules_with_slack[rule]
            input_traj_score = scores[rule]

            for item in self.trajs_tracked:
                item_scores = self.trajs_tracked[item]
                item_score = item_scores[rule]
                # Check if score of input is strictly better than current trajectory.
                if strictly_preceedes(rule, item_score, input_traj_score, slack):
                    # If so, check if it can be discarded.
                    if self.__discardable(scores, item_scores, rule):
                        return
                # Check if score of current trajectory is strictly better than input trajectory.
                elif strictly_preceedes(rule, input_traj_score, item_score, slack):
                    # If so, check if it can be discarded.
                    if self.__discardable(item_scores, scores, rule):
                        filter_index.append(item)
            self.trajs_tracked = {k: v for k, v in self.trajs_tracked.items() if k not in filter_index}
        self.trajs_tracked[egoname] = scores

    def __discardable(self, x, y, current_rule):
        """
        Checks whether a trajectory can be discarded, i.e. it will never be optimal.
        A trajectory X will never be optimal if there exists another trajectory Y for which
        there is a rule such that the score of Y is better than the score of X for that rule and they are
        not within the bound and the scores of Y for all other rules of higher priority are better compared to X.

        :param x: scores of trajectory to be possibly discarded
        :type x: Dict
        :param y: scores of trajectory to possibly discards x
        :param current_rule:
        :return:
        """
        for rule in self.rules_with_slack.keys():
            if (x[rule] == y[rule]) or strictly_preceedes(rule, y[rule], x[rule]):
                if current_rule == rule:
                    return True
            else:
                return False

    def get_optimal_trajs(self):
        optimal_set = self.trajs_tracked
        for rule in self.rules_with_slack.keys():
            slack = self.rules_with_slack[rule]
            if rule in self.decreasing:
                optimal_score = min(optimal_set.values(), key=lambda x: x[rule])[rule]
            else:
                optimal_score = max(optimal_set.values(), key=lambda x: x[rule])[rule]
            print("Optimal score for ", rule, optimal_score)
            # TODO remove print statements
            optimal_set = {k: v for k, v in optimal_set.items() if not strictly_preceedes(rule, optimal_score, v[rule], slack)}
            for item in optimal_set:
                print(item, ":", optimal_set[item][rule])
        return optimal_set

    def pop_best(self):
        best = self.get_optimal_trajs().copy()
        for k in best.keys():
            self.trajs_tracked.pop(k)
        return best


class LexicographicTracker(OptimalTrajectoryTracker):
    """
    Optimality is defined in the sense of lexicographic ordering where a trajectory X is deemed better than
    a trajectory Y if and only if there exists a rule for which the score of X is better than Y and for all
    rules of higher priority they are equivalent.
    """
    optimal_trajs: Dict[str, Dict[str, float]]
    rules: List[str]
    title: str

    def __init__(self, rules):
        self.optimal_trajs = dict()
        assert_valid_rules(rules)
        self.rules = rules
        self.title = 'Lexicographic Order'

    def digest_traj(self, egoname, scores):
        if not self.optimal_trajs:
            assert isinstance(scores, Dict)
            self.optimal_trajs[egoname] = scores
            return

        for rule in self.rules:
            input_traj_score = scores[rule]
            for item in self.optimal_trajs:
                item_scores = self.optimal_trajs[item]
                item_score = item_scores[rule]

                if strictly_preceedes(rule, item_score, input_traj_score):
                    return
                elif strictly_preceedes(rule, input_traj_score, item_score):
                    self.optimal_trajs.clear()
                    break
        self.optimal_trajs[egoname] = scores

    # TODO make abstract method
    def add_collection(self, trajs_collection: Dict[str, Dict[str, float]]):
        self.optimal_trajs = trajs_collection.copy()

    def get_optimal_trajs(self):
        return self.optimal_trajs

    def pop_best(self):
        optimal_set = self.optimal_trajs.copy()
        for rule in self.rules:
            if rule in self.decreasing:
                optimal_score = min(optimal_set.values(), key=lambda x: x[rule])[rule]
            else:
                optimal_score = max(optimal_set.values(), key=lambda x: x[rule])[rule]
            print("Optimal score for ", rule, optimal_score)
            # TODO remove print statements
            optimal_set = {k: v for k, v in optimal_set.items() if not strictly_preceedes(rule, optimal_score, v[rule])}
            for item in optimal_set:
                print(item, ":", optimal_set[item][rule])
        for k in optimal_set.keys():
            self.optimal_trajs.pop(k)

        return optimal_set


class ProductOrderTracker(OptimalTrajectoryTracker):
    """
    Optimality is defined in the sense of pareto optimality where a trajectory X is deemed better than
    a trajectory Y if and only if there exists no rule for which the score of Y is better than the score of X.
    """
    optimal_trajs: Dict[str, Dict[str, float]]
    rules: List[str]
    title: str

    def __init__(self, rules):
        self.optimal_trajs = dict()
        assert_valid_rules(rules)
        self.rules = rules
        self.title = 'Product Order'

    def digest_traj(self, egoname, scores):
        if not self.optimal_trajs:
            assert isinstance(scores, Dict)
            self.optimal_trajs[egoname] = scores
            return

        filter_index = []
        input_scores_as_array = np.fromiter(scores.values(), dtype=float)
        for item in self.optimal_trajs:
            item_scores_as_array = np.fromiter(self.optimal_trajs[item].values(), dtype=float)
            # TODO implement for max as well
            if np.all(np.less_equal(item_scores_as_array, input_scores_as_array)):
                return
            elif np.all(np.less_equal(input_scores_as_array, item_scores_as_array)):
                filter_index.append(item)

        self.optimal_trajs = {k: v for k, v in self.optimal_trajs.items() if k not in filter_index}
        self.optimal_trajs[egoname] = scores

    def add_collection(self, trajs_collection: Dict[str, Dict[str, float]]):
        self.optimal_trajs = trajs_collection.copy()

    def get_optimal_trajs(self):
        return self.optimal_trajs

    def pop_best(self):
        best = self.get_optimal_trajs().copy()
        for k in best.keys():
            self.optimal_trajs.pop(k)
        return best
