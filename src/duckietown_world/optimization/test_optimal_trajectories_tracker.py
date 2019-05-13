import unittest
import numpy as np
from collections import OrderedDict
from .optimal_trajectories_tracker import LexicographicSemiorderTracker, assert_valid_rules


class LexicographicSemiorderTest(unittest.TestCase):
    def testSimple(self):
        rules_list = OrderedDict([("Drivable areas", 0.05)])
        optimal_tracker = LexicographicSemiorderTracker(rules_list)
        self.assertTrue(assert_valid_rules(optimal_tracker.rules_with_slack))
        scores1 = {"Drivable areas": 0.5}
        scores2 = {"Drivable areas": 0.4}
        optimal_tracker.digest_traj("traj1", scores1)
        self.assertIsNotNone(optimal_tracker.trajs_tracked)
        optimal_tracker.digest_traj("traj2", scores2)
        self.assertEqual(len(optimal_tracker.trajs_tracked.keys()), 1)

    def testOrder(self):
        rules_list1 = {"Drivable areas": 0.05, "Survival time": 0.05}
        optimal_tracker1 = LexicographicSemiorderTracker(rules_list1)
        rules_list2 = {"Survival time": 0.05, "Drivable areas": 0.05}
        optimal_tracker2 = LexicographicSemiorderTracker(rules_list2)
        self.assertNotEqual(optimal_tracker1.rules_with_slack.popitem(), optimal_tracker2.rules_with_slack.popitem())

    def testTrajsTracked(self):
        rules_list = OrderedDict([("Drivable areas", 0.05), ("Survival time", 0.05)])
        optimal_tracker = LexicographicSemiorderTracker(rules_list)
        scores1 = {"Drivable areas": 0.5, "Survival time": 0.5}
        scores2 = {"Drivable areas": 0.5, "Survival time": 0.4}
        optimal_tracker.digest_traj("traj1", scores1)
        optimal_tracker.digest_traj("traj2", scores2)
        self.assertTrue("traj2" not in optimal_tracker.trajs_tracked.keys())
        scores3 = {"Drivable areas": 0.51, "Survival time": 0.4}
        optimal_tracker.digest_traj("traj3", scores3)
        self.assertTrue("traj3" in optimal_tracker.trajs_tracked.keys())

    def testGetOptimal(self):
        rules_list = OrderedDict([("Drivable areas", 0.05), ("Survival time", 0.05)])
        optimal_tracker = LexicographicSemiorderTracker(rules_list)
        scores1 = {"Drivable areas": 0.7, "Survival time": 0.8}
        scores2 = {"Drivable areas": 0.5, "Survival time": 0.4}
        optimal_tracker.digest_traj("traj1", scores1)
        optimal_tracker.digest_traj("traj2", scores2)
        optimal_set = optimal_tracker.get_optimal_trajs().keys()
        self.assertTrue("traj1" in optimal_set)
        self.assertTrue("traj2" not in optimal_set)


if __name__ == "__main__":
    unittest.main()
