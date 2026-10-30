import unittest

from explore_block13_gemm_pipeline import capacities, explore, graph, validate
from explore_instruction_overlap import schedule


class PipelineTests(unittest.TestCase):
    def test_full_weight_capacity_is_not_tile_capacity(self):
        report = explore(1280, 73856, 64)
        self.assertTrue(all(not stage["capacity_ok"] for stage in report["stages"]))
        self.assertEqual(capacities("MLP0"), (1345536, 18560))
        self.assertEqual(capacities("MLP2"), (1331712, 73856))

    def test_full_capacity_schedule_and_lifetime(self):
        report = explore(1345536, 73856, 64)
        self.assertTrue(all(stage["capacity_ok"] for stage in report["stages"]))
        for stage in report["stages"]:
            self.assertLessEqual(stage["overlap"]["cycles"], stage["serial"]["cycles"])
            validate(stage["overlap"]["tasks"], stage["stage"])
        tasks = {r["task"]: r for r in schedule(graph("QKV", "overlap", 64))}
        self.assertGreaterEqual(tasks["prep2"]["start"], tasks["gemm0"]["end"])
        self.assertEqual(tasks["prep0"]["start"], 0)
        self.assertEqual(tasks["weight"]["start"], 0)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            explore(0, 73856, 64)
        with self.assertRaises(ValueError):
            graph("QKV", "overlap", 0)


if __name__ == "__main__":
    unittest.main()
