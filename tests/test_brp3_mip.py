"""Testes pequenos e verificáveis da formulação BRP-III."""

import importlib.util
import unittest

from src.heuristics.lookahead_heuristic import solve_lookahead
from src.models.brp3_mip import (
    SolverUnavailableError,
    solve_brp3,
)
from src.problem import BlockRelocationProblem


def has_scipy_milp() -> bool:
    if importlib.util.find_spec("scipy") is None:
        return False
    from scipy.optimize import milp

    return callable(milp)


@unittest.skipUnless(
    has_scipy_milp() or importlib.util.find_spec("gurobipy") is not None,
    "BRP-III requer Gurobi com licença ou SciPy/HiGHS.",
)
class BRP3MIPTests(unittest.TestCase):
    def test_unblocked_instance_has_manual_optimum(self) -> None:
        problem = BlockRelocationProblem(
            stacks=[[1], [2]],
            retrieval_order=[1, 2],
            capacity=2,
        )

        result = solve_brp3(problem)

        self.assertEqual(result.status, "OPTIMAL")
        self.assertTrue(result.optimal)
        self.assertEqual(result.objective, 2)
        self.assertEqual(result.relocation_count, 0)
        self.assertEqual(result.mip_gap, 0)
        self.assertEqual(
            [(move.period, move.action, move.container) for move in result.moves],
            [(1, "retrieve", 1), (2, "retrieve", 2)],
        )
        self.assertTrue(problem.validate())
        self.assertEqual(problem.retrieved, [])

    def test_one_blocker_requires_one_relocation_and_matches_la1(self) -> None:
        stacks = [[1, 3], [2], []]
        order = [1, 2, 3]
        capacity = 3
        problem = BlockRelocationProblem(stacks, order, capacity)
        heuristic_problem = BlockRelocationProblem(stacks, order, capacity)

        result = solve_brp3(problem)
        heuristic = solve_lookahead(heuristic_problem, 1)

        self.assertEqual(result.status, "OPTIMAL")
        self.assertTrue(result.optimal)
        self.assertEqual(result.objective, 4)
        self.assertEqual(result.relocation_count, 1)
        self.assertEqual(result.relocation_count, heuristic.relocation_count)
        self.assertEqual(
            [(move.action, move.container) for move in result.moves],
            [
                ("relocate", 3),
                ("retrieve", 1),
                ("retrieve", 2),
                ("retrieve", 3),
            ],
        )
        self.assertEqual(result.final_stacks, ((), (), ()))
        self.assertEqual(problem.stacks, stacks)
        self.assertEqual(problem.retrieved, [])

    def test_explicit_larger_horizon_preserves_optimum(self) -> None:
        problem = BlockRelocationProblem(
            stacks=[[1, 3], [2], []],
            retrieval_order=[1, 2, 3],
            capacity=3,
        )

        result = solve_brp3(problem, horizon=6)

        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.objective, 4)
        self.assertEqual(result.relocation_count, 1)

    def test_infeasible_full_bay_is_reported_by_solver(self) -> None:
        problem = BlockRelocationProblem(
            stacks=[[1, 2], [3, 4]],
            retrieval_order=[1, 2, 3, 4],
            capacity=2,
        )

        result = solve_brp3(problem, horizon=6)

        self.assertEqual(result.status, "INFEASIBLE")
        self.assertFalse(result.optimal)
        self.assertIsNone(result.objective)
        self.assertIsNone(result.relocation_count)
        self.assertEqual(result.moves, ())

    def test_adjacent_stack_safety_constraints_are_enabled_by_default(self) -> None:
        problem = BlockRelocationProblem(
            stacks=[[1, 2, 3], []],
            retrieval_order=[1, 2, 3],
            capacity=4,
        )

        result = solve_brp3(problem, horizon=4)

        self.assertEqual(result.status, "INFEASIBLE")

    def test_paper_experiment_mode_can_omit_safety_constraints(self) -> None:
        problem = BlockRelocationProblem(
            stacks=[[1, 2, 3], []],
            retrieval_order=[1, 2, 3],
            capacity=4,
        )

        result = solve_brp3(
            problem,
            horizon=5,
            include_safety_constraints=False,
        )

        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.objective, 5)
        self.assertEqual(result.relocation_count, 2)

    def test_empty_instance_and_too_short_horizon_are_rejected(self) -> None:
        empty = BlockRelocationProblem([[], []], [], capacity=2)
        with self.assertRaisesRegex(ValueError, "ao menos um contêiner"):
            solve_brp3(empty)

        problem = BlockRelocationProblem([[1], [2]], [1, 2], capacity=2)
        with self.assertRaisesRegex(ValueError, "pelo menos o número"):
            solve_brp3(problem, horizon=1)

    def test_gurobi_request_reports_missing_package(self) -> None:
        if importlib.util.find_spec("gurobipy") is not None:
            self.skipTest("gurobipy está instalado neste ambiente.")
        problem = BlockRelocationProblem([[1], [2]], [1, 2], capacity=2)
        with self.assertRaisesRegex(SolverUnavailableError, "gurobipy não está instalado"):
            solve_brp3(problem, backend="gurobi")


if __name__ == "__main__":
    unittest.main()
