from abc import ABCMeta, abstractmethod
from typing import List, Dict

__all__ = [
    'OptimalTrajectoryTracker',
    'assert_valid_rules',
    'strictly_precedes',
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

    trajs_tracked: Dict[str, Dict[str, float]]
    title: str

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

    @abstractmethod
    def pop_best(self):
        """

        :return:
        """

    def add_collection(self, trajs_collection: Dict[str, Dict[str, float]]):
        self.trajs_tracked = trajs_collection.copy()


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


def strictly_precedes(rule: str, score_x: float, score_y: float, slack=0.0):
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

    rules_with_slack: Dict[str, float]

    def __init__(self, rules_with_slack: Dict[str, float]):
        self.trajs_tracked = dict()
        assert_valid_rules(rules_with_slack.keys())
        self.rules_with_slack = rules_with_slack
        self.title = 'Lexicographic Semiorder'

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
                if strictly_precedes(rule, item_score, input_traj_score, slack):
                    # If so, check if it can be discarded.
                    if self.__discardable(scores, item_scores, rule):
                        return
                # Check if score of current trajectory is strictly better than input trajectory.
                elif strictly_precedes(rule, input_traj_score, item_score, slack):
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
            if (x[rule] == y[rule]) or strictly_precedes(rule, y[rule], x[rule]):
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
            # print("Optimal score for ", rule, optimal_score)
            optimal_set = {k: v for k, v in optimal_set.items() if not strictly_precedes(rule, optimal_score, v[rule], slack)}
            # for item in optimal_set:
                # print(item, ":", optimal_set[item][rule])
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
    rules: List[str]

    def __init__(self, rules):
        self.trajs_tracked = dict()
        assert_valid_rules(rules)
        self.rules = rules
        self.title = 'Lexicographic Order'

    def digest_traj(self, egoname, scores):
        if not self.trajs_tracked:
            assert isinstance(scores, Dict)
            self.trajs_tracked[egoname] = scores
            return

        for rule in self.rules:
            input_traj_score = scores[rule]
            for item in self.trajs_tracked:
                item_scores = self.trajs_tracked[item]
                item_score = item_scores[rule]

                if strictly_precedes(rule, item_score, input_traj_score):
                    return
                elif strictly_precedes(rule, input_traj_score, item_score):
                    self.trajs_tracked.clear()
                    break
        self.trajs_tracked[egoname] = scores

    def get_optimal_trajs(self):
        return self.trajs_tracked

    def pop_best(self):
        optimal_set = self.trajs_tracked.copy()
        for rule in self.rules:
            if rule in self.decreasing:
                optimal_score = min(optimal_set.values(), key=lambda x: x[rule])[rule]
            else:
                optimal_score = max(optimal_set.values(), key=lambda x: x[rule])[rule]
            optimal_set = {k: v for k, v in optimal_set.items() if not strictly_precedes(rule, optimal_score, v[rule])}
            for item in optimal_set:
                print(item, ":", optimal_set[item][rule])
        for k in optimal_set.keys():
            self.trajs_tracked.pop(k)

        return optimal_set


class ProductOrderTracker(OptimalTrajectoryTracker):
    """
    Optimality is defined in the sense of pareto optimality where a trajectory X is deemed better than
    a trajectory Y if and only if there exists no rule for which the score of Y is better than the score of X.
    """
    rules: List[str]

    def __init__(self, rules):
        self.trajs_tracked = dict()
        assert_valid_rules(rules)
        self.rules = rules
        self.title = 'Product Order'

    def digest_traj(self, egoname, scores):
        if not self.trajs_tracked:
            assert isinstance(scores, Dict)
            self.trajs_tracked[egoname] = scores
            return

        filter_index = []
        for item in self.trajs_tracked:
            item_scores = self.trajs_tracked[item]

            if self.compare(item_scores, scores):
                return
            elif self.compare(scores, item_scores):
                filter_index.append(item)

        self.trajs_tracked = {k: v for k, v in self.trajs_tracked.items() if k not in filter_index}
        self.trajs_tracked[egoname] = scores

    def compare_scores(self, x: Dict[str, float], y: Dict[str, float]):
        """
        :param x: scores of x
        :param y: scores of y
        :return: True if x strictly precedes y in all dimensions
        """
        strictly_preceding = False
        for rule in self.rules:
            if strictly_precedes(rule, x[rule], y[rule]):
                strictly_preceding = True
                continue
            elif strictly_precedes(rule, y[rule], x[rule]):
                return False
        return strictly_preceding

    def get_optimal_trajs(self):
        return self.trajs_tracked

    def pop_best(self):
        best = self.get_optimal_trajs().copy()
        for k in best.keys():
            self.trajs_tracked.pop(k)
        return best
