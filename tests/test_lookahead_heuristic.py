"""Testes da heurística Extended Look-Ahead LA-N."""

import unittest

from src.heuristics.lookahead_heuristic import solve_lookahead
from src.problem import BlockRelocationProblem, small_instance


class LookaheadHeuristicTests(unittest.TestCase):
    def replay_and_validate(self, initial: BlockRelocationProblem, moves: tuple) -> None:
        replay = BlockRelocationProblem(
            stacks=initial.stacks,
            retrieval_order=initial.retrieval_order,
            capacity=initial.capacity,
        )
        for move in moves:
            if move.action == "relocate":
                self.assertIsNotNone(move.destination)
                self.assertEqual(
                    replay.relocate(move.source, move.destination), move.container
                )
            else:
                self.assertEqual(replay.retrieve(move.source), move.container)
            self.assertTrue(replay.validate())
            self.assertTrue(
                all(replay.height(i) <= replay.capacity for i in range(replay.stack_count))
            )
        self.assertTrue(replay.is_complete)
        self.assertEqual(replay.retrieved, list(replay.retrieval_order))

    def test_small_instance_for_multiple_lookaheads(self) -> None:
        for n in (1, 2, 3):
            with self.subTest(n=n):
                problem = small_instance()
                initial = BlockRelocationProblem(
                    problem.stacks, problem.retrieval_order, problem.capacity
                )
                result = solve_lookahead(problem, n)
                self.assertEqual(result.lookahead, n)
                self.assertTrue(problem.is_complete)
                self.assertTrue(problem.validate())
                self.assertEqual(result.relocation_count, problem.relocation_count)
                self.replay_and_validate(initial, result.moves)

    def test_extended_lookahead_makes_a_cleaning_move(self) -> None:
        initial = BlockRelocationProblem(
            stacks=[[2, 4], [3, 6], [1, 5]],
            retrieval_order=[1, 2, 3, 4, 5, 6],
            capacity=3,
        )

        la1_problem = BlockRelocationProblem(
            initial.stacks, initial.retrieval_order, initial.capacity
        )
        la1 = solve_lookahead(la1_problem, 1)
        self.assertFalse(any(move.cleaning for move in la1.moves))

        la2_problem = BlockRelocationProblem(
            initial.stacks, initial.retrieval_order, initial.capacity
        )
        la2 = solve_lookahead(la2_problem, 2)
        cleaning_moves = [move for move in la2.moves if move.cleaning]
        self.assertEqual(
            [(move.container, move.source, move.destination) for move in cleaning_moves],
            [(5, 1, 2)],
        )
        self.assertEqual(la1.relocation_count, 5)
        self.assertEqual(la2.relocation_count, 4)
        self.replay_and_validate(initial, la1.moves)
        self.replay_and_validate(initial, la2.moves)

    def test_large_n_is_capped_by_remaining_containers(self) -> None:
        problem = small_instance()
        result = solve_lookahead(problem, 100)
        self.assertEqual(result.lookahead, 100)
        self.assertTrue(problem.is_complete)
        self.assertTrue(problem.validate())

    def test_nonpositive_or_noninteger_n_is_rejected(self) -> None:
        for n in (0, -1, 1.5, True):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    solve_lookahead(small_instance(), n)

    def test_infeasible_full_bay_is_reported(self) -> None:
        problem = BlockRelocationProblem(
            stacks=[[1, 2], [3, 4]],
            retrieval_order=[1, 2, 3, 4],
            capacity=2,
        )
        with self.assertRaisesRegex(ValueError, "não há pilha de destino"):
            solve_lookahead(problem, 1)


if __name__ == "__main__":
    unittest.main()
