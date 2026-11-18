"""Formulação BRP-III de Petering e Hussein (2013).

Índices Python começam em zero. Os índices matemáticos ``c=1..C``,
``s=1..S`` e ``t=1..W`` correspondem, respectivamente, à posição na ordem
de retirada, ao índice da pilha e ao período da solução.

A formulação implementada é a das Tabelas 1–4 do artigo:

    min  sum_t t T[C,t]

    (1a-c) R[c,s,t] <= R[s,t]; R[c,s,t] <= M[c,t];
           R[c,s,t] >= R[s,t] + M[c,t] - 1
    (1d-f) 0 <= R[c,s,t] <= 1; sum_c R[c,s,t] = R[s,t];
           sum_s R[c,s,t] = M[c,t]
    (2a-c) P[c,s,t] <= P[s,t]; P[c,s,t] <= M[c,t];
           P[c,s,t] >= P[s,t] + M[c,t] - 1
    (2d-f) 0 <= P[c,s,t] <= 1; sum_c P[c,s,t] = P[s,t];
           sum_s P[c,s,t] <= M[c,t]
    (3a-c) X[c,s,1] = initialSetup[c,s];
           X[c,s,t+1] = X[c,s,t] + P[c,s,t] - R[c,s,t];
           0 <= X[c,s,t] <= 1
    (4a-c) B[c,1] = initialBury[c];
           B[c,t+1] = B[c,t] + F[c,t] - C[c,t]; B[c,t] >= 0
    (5)   sum_s X[c,s,t] <= 1
    (6)   sum_c X[c,s,t] <= mxHeight
    (7-8) h[s,t] <= 2 + h[s+1,t]; h[s,t] >= h[s+1,t] - 2
    (9)   B[c,t] <= mxHeight
    (10)  sum_t M[c,t] >= 1
    (11)  sum_c M[c,t] <= 1
    (12)  sum_c M[c,t] = 1, t=1..C
    (13-14) sum_c C[c,t] <= mxHeight; sum_c F[c,t] <= mxHeight
    (15)  sum_t (C[c,t] - F[c,t]) = initialBury[c]
    (16)  sum_c T[c,t] <= 1
    (17)  sum_t T[c,t] = 1
    (18)  sum_t t T[c+1,t] >= 1 + sum_t t T[c,t], c<C
    (19)  sum_s R[s,t] <= 1
    (20)  sum_s R[s,t] = 1, t=1..C
    (21)  sum_s P[s,t] <= sum_s R[s,t]
    (22)  P[s,t] + R[s,t] <= 1
    (23)  sum_s P[c,s,t] <= sum_s R[c,s,t]
    (24)  R[c,s,t] <= X[c,s,t]
    (25)  (1-M[c,t]) (mxHeight-1) >= B[c,t]-1
    (26)  T[c,t] <= M[c,t]
    (27)  T[c,t] = sum_s R[c,s,t] - sum_s P[c,s,t]
    (28a-c) C[c,t] = X[c,s,t] AND R[s,t]
    (29a-c) F[c,t] = X[c,s,t+1] AND P[s,t]

Here h[s,t] = sum_c X[c,s,t]. X and B are defined for t=1..W+1;
the remaining variables are defined for t=1..W. M, C, F, T, R[s,t],
P[s,t] are binary. X, R[c,s,t], and P[c,s,t] are continuous in [0,1].
B is continuous and nonnegative; its linking constraints make it integer.

Constraints (7)–(8) are the paper's starred adjacent-stack safety
constraints. They are enabled by default; ``include_safety_constraints=False``
omits only these two groups, as in the computational experiments described
in Section 3.2 of the article.

Gurobi is preferred when installed. Otherwise ``backend="auto"`` uses
SciPy's ``milp`` interface to the open-source HiGHS solver. Neither backend
silently substitutes another mathematical formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from time import perf_counter
from typing import Literal

from src.problem import BlockRelocationProblem, ContainerId


Backend = Literal["auto", "gurobi", "scipy"]
Sense = Literal["<=", ">=", "="]


class SolverUnavailableError(RuntimeError):
    """O backend solicitado não está instalado ou não pode ser carregado."""


@dataclass(frozen=True)
class MIPMove:
    """Movimento decodificado de um período; pilhas usam índices a partir de zero."""

    period: int
    action: Literal["relocate", "retrieve"]
    container: ContainerId
    source: int
    destination: int | None = None


@dataclass(frozen=True)
class BRP3Result:
    """Resultado do solver, distinguindo incumbente viável de ótimo provado."""

    status: str
    optimal: bool
    objective: float | None
    relocation_count: int | None
    runtime_seconds: float
    solver_runtime_seconds: float
    mip_gap: float | None
    backend: str
    moves: tuple[MIPMove, ...]
    final_stacks: tuple[tuple[ContainerId, ...], ...] | None
    message: str


@dataclass(frozen=True)
class _Variable:
    name: str
    lower: float
    upper: float
    integer: bool


@dataclass(frozen=True)
class _Constraint:
    name: str
    coefficients: dict[int, float]
    sense: Sense
    rhs: float


class _LinearModel:
    """Armazena uma representação linear comum aos backends Gurobi e SciPy."""

    def __init__(self) -> None:
        self.variables: list[_Variable] = []
        self.constraints: list[_Constraint] = []
        self.objective: dict[int, float] = {}

    def add_variable(
        self, name: str, lower: float, upper: float, *, integer: bool = False
    ) -> int:
        index = len(self.variables)
        self.variables.append(_Variable(name, lower, upper, integer))
        return index

    def add_constraint(
        self,
        name: str,
        coefficients: dict[int, float],
        sense: Sense,
        rhs: float,
    ) -> None:
        cleaned = {
            index: coefficient
            for index, coefficient in coefficients.items()
            if coefficient != 0
        }
        self.constraints.append(_Constraint(name, cleaned, sense, rhs))


def _combine(*parts: tuple[dict[int, float], float]) -> dict[int, float]:
    result: dict[int, float] = {}
    for coefficients, multiplier in parts:
        for index, coefficient in coefficients.items():
            result[index] = result.get(index, 0.0) + multiplier * coefficient
    return result


def _sum_variables(indices: list[int]) -> dict[int, float]:
    return {index: 1.0 for index in indices}


def _build_model(
    initial_stacks: list[list[ContainerId]],
    container_order: tuple[ContainerId, ...],
    capacity: int,
    horizon: int,
    *,
    include_safety_constraints: bool,
) -> tuple[_LinearModel, dict[str, dict[tuple[int, ...], int]], list[int]]:
    """Constrói literalmente as variáveis e restrições da formulação BRP-III."""
    container_count = len(container_order)
    stack_count = len(initial_stacks)
    model = _LinearModel()
    variables: dict[str, dict[tuple[int, ...], int]] = {
        name: {}
        for name in ("X", "B", "M", "C", "F", "T", "R", "P", "Rc", "Pc")
    }

    def add(
        family: str,
        key: tuple[int, ...],
        lower: float,
        upper: float,
        *,
        integer: bool = False,
    ) -> int:
        index = model.add_variable(
            f"{family}[{','.join(str(item + 1) for item in key)}]",
            lower,
            upper,
            integer=integer,
        )
        variables[family][key] = index
        return index

    for c in range(container_count):
        for s in range(stack_count):
            for t in range(horizon + 1):
                add("X", (c, s, t), 0.0, 1.0)
        for t in range(horizon + 1):
            add("B", (c, t), 0.0, inf)
        for family in ("M", "C", "F", "T"):
            for t in range(horizon):
                add(family, (c, t), 0.0, 1.0, integer=True)

    for s in range(stack_count):
        for t in range(horizon):
            add("R", (s, t), 0.0, 1.0, integer=True)
            add("P", (s, t), 0.0, 1.0, integer=True)

    for c in range(container_count):
        for s in range(stack_count):
            for t in range(horizon):
                add("Rc", (c, s, t), 0.0, 1.0)
                add("Pc", (c, s, t), 0.0, 1.0)

    rank = {container: c for c, container in enumerate(container_order)}
    initial_setup: dict[tuple[int, int], int] = {}
    initial_bury = [0] * container_count
    for s, stack in enumerate(initial_stacks):
        for position, container in enumerate(stack):
            c = rank[container]
            initial_setup[c, s] = 1
            initial_bury[c] = len(stack) - position

    def eq(name: str, coefficients: dict[int, float], rhs: float = 0.0) -> None:
        model.add_constraint(name, coefficients, "=", rhs)

    def le(name: str, coefficients: dict[int, float], rhs: float = 0.0) -> None:
        model.add_constraint(name, coefficients, "<=", rhs)

    def ge(name: str, coefficients: dict[int, float], rhs: float = 0.0) -> None:
        model.add_constraint(name, coefficients, ">=", rhs)

    # (1a)-(1f): variáveis de remoção R[c,s,t] e R[s,t].
    for c in range(container_count):
        for s in range(stack_count):
            for t in range(horizon):
                rc, rs, moved = (
                    variables["Rc"][c, s, t],
                    variables["R"][s, t],
                    variables["M"][c, t],
                )
                le(f"1a[{c},{s},{t}]", {rc: 1, rs: -1})
                le(f"1b[{c},{s},{t}]", {rc: 1, moved: -1})
                ge(f"1c[{c},{s},{t}]", {rc: 1, rs: -1, moved: -1}, -1)
    for s in range(stack_count):
        for t in range(horizon):
            eq(
                f"1e[{s},{t}]",
                _combine(
                    (_sum_variables([variables["Rc"][c, s, t] for c in range(container_count)]), 1),
                    ({variables["R"][s, t]: 1}, -1),
                ),
            )
    for c in range(container_count):
        for t in range(horizon):
            eq(
                f"1f[{c},{t}]",
                _combine(
                    (_sum_variables([variables["Rc"][c, s, t] for s in range(stack_count)]), 1),
                    ({variables["M"][c, t]: 1}, -1),
                ),
            )

    # (2a)-(2f): variáveis de colocação P[c,s,t] e P[s,t].
    for c in range(container_count):
        for s in range(stack_count):
            for t in range(horizon):
                pc, ps, moved = (
                    variables["Pc"][c, s, t],
                    variables["P"][s, t],
                    variables["M"][c, t],
                )
                le(f"2a[{c},{s},{t}]", {pc: 1, ps: -1})
                le(f"2b[{c},{s},{t}]", {pc: 1, moved: -1})
                ge(f"2c[{c},{s},{t}]", {pc: 1, ps: -1, moved: -1}, -1)
    for s in range(stack_count):
        for t in range(horizon):
            eq(
                f"2e[{s},{t}]",
                _combine(
                    (_sum_variables([variables["Pc"][c, s, t] for c in range(container_count)]), 1),
                    ({variables["P"][s, t]: 1}, -1),
                ),
            )
    for c in range(container_count):
        for t in range(horizon):
            le(
                f"2f[{c},{t}]",
                _combine(
                    (_sum_variables([variables["Pc"][c, s, t] for s in range(stack_count)]), 1),
                    ({variables["M"][c, t]: 1}, -1),
                ),
            )

    # (3a)-(3c): localização no começo de cada período.
    for c in range(container_count):
        for s in range(stack_count):
            eq(
                f"3a[{c},{s}]",
                {variables["X"][c, s, 0]: 1},
                initial_setup.get((c, s), 0),
            )
            for t in range(horizon):
                eq(
                    f"3b[{c},{s},{t}]",
                    {
                        variables["X"][c, s, t + 1]: 1,
                        variables["X"][c, s, t]: -1,
                        variables["Pc"][c, s, t]: -1,
                        variables["Rc"][c, s, t]: 1,
                    },
                )

    # (4a)-(4c): número de contêineres cobrindo cada contêiner.
    for c in range(container_count):
        eq(f"4a[{c}]", {variables["B"][c, 0]: 1}, initial_bury[c])
        for t in range(horizon):
            eq(
                f"4b[{c},{t}]",
                {
                    variables["B"][c, t + 1]: 1,
                    variables["B"][c, t]: -1,
                    variables["F"][c, t]: -1,
                    variables["C"][c, t]: 1,
                },
            )
    for c in range(container_count):
        for t in range(horizon + 1):
            ge(f"4c[{c},{t}]", {variables["B"][c, t]: 1}, 0)

    # (5)-(9): capacidade, posição única e segurança entre pilhas adjacentes.
    for c in range(container_count):
        for t in range(horizon + 1):
            le(
                f"5[{c},{t}]",
                _sum_variables([variables["X"][c, s, t] for s in range(stack_count)]),
                1,
            )
    for s in range(stack_count):
        for t in range(horizon + 1):
            le(
                f"6[{s},{t}]",
                _sum_variables([variables["X"][c, s, t] for c in range(container_count)]),
                capacity,
            )
    if include_safety_constraints:
        for s in range(stack_count - 1):
            for t in range(horizon + 1):
                heights = {
                    variables["X"][c, s, t]: 1 for c in range(container_count)
                }
                for c in range(container_count):
                    heights[variables["X"][c, s + 1, t]] = -1
                le(f"7[{s},{t}]", heights, 2)
                ge(f"8[{s},{t}]", heights, -2)
    for c in range(container_count):
        for t in range(horizon + 1):
            le(f"9[{c},{t}]", {variables["B"][c, t]: 1}, capacity)

    # (10)-(15): movimentações e variação da profundidade dos contêineres.
    for c in range(container_count):
        ge(
            f"10[{c}]",
            _sum_variables([variables["M"][c, t] for t in range(horizon)]),
            1,
        )
    for t in range(horizon):
        le(
            f"11[{t}]",
            _sum_variables([variables["M"][c, t] for c in range(container_count)]),
            1,
        )
        if t < container_count:
            eq(
                f"12[{t}]",
                _sum_variables([variables["M"][c, t] for c in range(container_count)]),
                1,
            )
        le(
            f"13[{t}]",
            _sum_variables([variables["C"][c, t] for c in range(container_count)]),
            capacity,
        )
        le(
            f"14[{t}]",
            _sum_variables([variables["F"][c, t] for c in range(container_count)]),
            capacity,
        )
    for c in range(container_count):
        eq(
            f"15[{c}]",
            _combine(
                (_sum_variables([variables["C"][c, t] for t in range(horizon)]), 1),
                (_sum_variables([variables["F"][c, t] for t in range(horizon)]), -1),
            ),
            initial_bury[c],
        )

    # (16)-(18): remoção definitiva e ordem de retirada.
    for t in range(horizon):
        le(
            f"16[{t}]",
            _sum_variables([variables["T"][c, t] for c in range(container_count)]),
            1,
        )
    for c in range(container_count):
        eq(
            f"17[{c}]",
            _sum_variables([variables["T"][c, t] for t in range(horizon)]),
            1,
        )
    for c in range(container_count - 1):
        order = {
            variables["T"][c + 1, t]: float(t + 1) for t in range(horizon)
        }
        for t in range(horizon):
            order[variables["T"][c, t]] = order.get(variables["T"][c, t], 0) - (t + 1)
        ge(f"18[{c}]", order, 1)

    # (19)-(23): uma remoção por turno, colocação acoplada e pilha distinta.
    for t in range(horizon):
        stack_removals = _sum_variables(
            [variables["R"][s, t] for s in range(stack_count)]
        )
        le(f"19[{t}]", stack_removals, 1)
        if t < container_count:
            eq(f"20[{t}]", stack_removals, 1)
        le(
            f"21[{t}]",
            _combine(
                (_sum_variables([variables["P"][s, t] for s in range(stack_count)]), 1),
                (stack_removals, -1),
            ),
        )
        for s in range(stack_count):
            le(
                f"22[{s},{t}]",
                {variables["P"][s, t]: 1, variables["R"][s, t]: 1},
                1,
            )
    for c in range(container_count):
        for t in range(horizon):
            le(
                f"23[{c},{t}]",
                _combine(
                    (_sum_variables([variables["Pc"][c, s, t] for s in range(stack_count)]), 1),
                    (_sum_variables([variables["Rc"][c, s, t] for s in range(stack_count)]), -1),
                ),
            )

    # (24)-(27): disponibilidade no topo e ligação com as retiradas finais.
    for c in range(container_count):
        for s in range(stack_count):
            for t in range(horizon):
                le(
                    f"24[{c},{s},{t}]",
                    {
                        variables["Rc"][c, s, t]: 1,
                        variables["X"][c, s, t]: -1,
                    },
                )
    for c in range(container_count):
        for t in range(horizon):
            le(
                f"25[{c},{t}]",
                {
                    variables["B"][c, t]: 1,
                    variables["M"][c, t]: capacity - 1,
                },
                capacity,
            )
            le(
                f"26[{c},{t}]",
                {variables["T"][c, t]: 1, variables["M"][c, t]: -1},
            )
            eq(
                f"27[{c},{t}]",
                _combine(
                    ({variables["T"][c, t]: 1}, 1),
                    (_sum_variables([variables["Rc"][c, s, t] for s in range(stack_count)]), -1),
                    (_sum_variables([variables["Pc"][c, s, t] for s in range(stack_count)]), 1),
                ),
            )

    # (28)-(29): C[c,t] = X[c,s,t] AND R[s,t];
    # F[c,t] = X[c,s,t+1] AND P[s,t].
    for c in range(container_count):
        for s in range(stack_count):
            for t in range(horizon):
                closer, x_now, removed = (
                    variables["C"][c, t],
                    variables["X"][c, s, t],
                    variables["R"][s, t],
                )
                le(f"28a[{c},{s},{t}]", {closer: 1, removed: 1, x_now: -1}, 1)
                le(f"28b[{c},{s},{t}]", {x_now: 1, closer: 1, removed: -1}, 1)
                le(f"28c[{c},{s},{t}]", {removed: 1, x_now: 1, closer: -1}, 1)

                farther, x_next, placed = (
                    variables["F"][c, t],
                    variables["X"][c, s, t + 1],
                    variables["P"][s, t],
                )
                le(f"29a[{c},{s},{t}]", {farther: 1, placed: 1, x_next: -1}, 1)
                le(f"29b[{c},{s},{t}]", {x_next: 1, farther: 1, placed: -1}, 1)
                le(f"29c[{c},{s},{t}]", {placed: 1, x_next: 1, farther: -1}, 1)

    for t in range(horizon):
        model.objective[variables["T"][container_count - 1, t]] = float(t + 1)

    return model, variables, initial_bury


def _prepare_instance(
    problem: BlockRelocationProblem, horizon: int | None
) -> tuple[list[list[ContainerId]], tuple[ContainerId, ...], int]:
    """Valida o estado atual e escolhe W segundo a convenção do artigo."""
    problem.validate()
    container_order = problem.retrieval_order[len(problem.retrieved) :]
    if not container_order:
        raise ValueError("BRP-III requer ao menos um contêiner ainda não retirado.")
    if isinstance(problem.capacity, bool) or not isinstance(problem.capacity, int):
        raise ValueError("mxHeight deve ser um inteiro.")
    if problem.capacity <= 1:
        raise ValueError("A formulação publicada requer mxHeight > 1.")

    initial_stacks = [list(stack) for stack in problem.stacks]
    if horizon is None:
        from src.heuristics.lookahead_heuristic import solve_lookahead

        upper_bound_problem = BlockRelocationProblem(
            initial_stacks, container_order, problem.capacity
        )
        upper_bound = solve_lookahead(upper_bound_problem, 1)
        horizon = len(upper_bound.moves)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < len(
        container_order
    ):
        raise ValueError(
            "W deve ser um inteiro e ser pelo menos o número de contêineres "
            "restantes (restrições 12 e 20)."
        )
    return initial_stacks, tuple(container_order), horizon


def _select_backend(backend: Backend) -> str:
    if backend not in ("auto", "gurobi", "scipy"):
        raise ValueError("backend deve ser 'auto', 'gurobi' ou 'scipy'.")
    if backend in ("auto", "gurobi"):
        try:
            import gurobipy  # noqa: F401
        except ImportError as error:
            if backend == "gurobi":
                raise SolverUnavailableError(
                    "gurobipy não está instalado. Instale o pacote e configure "
                    "uma licença Gurobi válida."
                ) from error
        else:
            return "gurobi"
    try:
        import scipy.optimize  # noqa: F401
    except ImportError as error:
        raise SolverUnavailableError(
            "Nenhum backend disponível: instale gurobipy com licença válida ou "
            "SciPy com scipy.optimize.milp (HiGHS)."
        ) from error
    return "scipy"


def _solve_gurobi(
    model: _LinearModel,
    *,
    time_limit: float | None,
    mip_gap: float | None,
    output: bool,
) -> tuple[list[float] | None, str, float | None, float, float | None, str]:
    import gurobipy as gp

    optimizer = gp.Model("BRP-III")
    optimizer.Params.OutputFlag = int(output)
    if time_limit is not None:
        optimizer.Params.TimeLimit = time_limit
    if mip_gap is not None:
        optimizer.Params.MIPGap = mip_gap

    variables = []
    for variable in model.variables:
        variables.append(
            optimizer.addVar(
                lb=variable.lower,
                ub=variable.upper,
                vtype=gp.GRB.BINARY if variable.integer else gp.GRB.CONTINUOUS,
                name=variable.name,
            )
        )
    optimizer.update()
    for constraint in model.constraints:
        expression = gp.quicksum(
            coefficient * variables[index]
            for index, coefficient in constraint.coefficients.items()
        )
        if constraint.sense == "=":
            optimizer.addConstr(expression == constraint.rhs, name=constraint.name)
        elif constraint.sense == "<=":
            optimizer.addConstr(expression <= constraint.rhs, name=constraint.name)
        else:
            optimizer.addConstr(expression >= constraint.rhs, name=constraint.name)
    optimizer.setObjective(
        gp.quicksum(
            coefficient * variables[index]
            for index, coefficient in model.objective.items()
        ),
        gp.GRB.MINIMIZE,
    )

    start = perf_counter()
    optimizer.optimize()
    elapsed = perf_counter() - start
    status_map = {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    status = status_map.get(optimizer.Status, f"GUROBI_STATUS_{optimizer.Status}")
    solution = (
        [variable.X for variable in variables] if optimizer.SolCount > 0 else None
    )
    objective = float(optimizer.ObjVal) if solution is not None else None
    gap = float(optimizer.MIPGap) if solution is not None else None
    message = f"Gurobi status: {status}."
    return solution, status, objective, elapsed, gap, message


def _solve_scipy(
    model: _LinearModel,
    *,
    time_limit: float | None,
    mip_gap: float | None,
    output: bool,
) -> tuple[list[float] | None, str, float | None, float, float | None, str]:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower_rows: list[float] = []
    upper_rows: list[float] = []
    for row_index, constraint in enumerate(model.constraints):
        for column, coefficient in constraint.coefficients.items():
            row_indices.append(row_index)
            column_indices.append(column)
            values.append(coefficient)
        lower_rows.append(
            constraint.rhs if constraint.sense in (">=", "=") else -np.inf
        )
        upper_rows.append(
            constraint.rhs if constraint.sense in ("<=", "=") else np.inf
        )

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(model.constraints), len(model.variables)),
    ).tocsr()
    objective = np.zeros(len(model.variables))
    for index, coefficient in model.objective.items():
        objective[index] = coefficient
    lower_bounds = np.array([variable.lower for variable in model.variables])
    upper_bounds = np.array([variable.upper for variable in model.variables])
    integrality = np.array([int(variable.integer) for variable in model.variables])
    options: dict[str, float | bool] = {"disp": output}
    if time_limit is not None:
        options["time_limit"] = time_limit
    if mip_gap is not None:
        options["mip_rel_gap"] = mip_gap

    start = perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix, lower_rows, upper_rows),
        options=options,
    )
    elapsed = perf_counter() - start
    status_map = {
        0: "OPTIMAL",
        1: "LIMIT_REACHED",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "SOLVER_ERROR",
    }
    status = status_map.get(int(result.status), f"SCIPY_STATUS_{result.status}")
    solution = result.x.tolist() if result.x is not None else None
    objective_value = float(result.fun) if result.fun is not None else None
    raw_gap = getattr(result, "mip_gap", None)
    gap = float(raw_gap) if raw_gap is not None else None
    return solution, status, objective_value, elapsed, gap, str(result.message)


def _validate_linear_solution(
    model: _LinearModel, solution: list[float], tolerance: float = 1e-5
) -> None:
    """Confere limites, integralidade e todas as restrições antes da extração."""
    if len(solution) != len(model.variables):
        raise RuntimeError("O solver retornou um vetor de solução com dimensão inválida.")
    for index, (value, variable) in enumerate(zip(solution, model.variables)):
        if not variable.lower - tolerance <= value <= variable.upper + tolerance:
            raise RuntimeError(
                f"Solução viola os limites da variável {variable.name}: {value}."
            )
        if variable.integer and abs(value - round(value)) > tolerance:
            raise RuntimeError(
                f"Solução viola integralidade da variável {variable.name}: {value}."
            )
    for constraint in model.constraints:
        lhs = sum(
            coefficient * solution[index]
            for index, coefficient in constraint.coefficients.items()
        )
        violated = (
            abs(lhs - constraint.rhs) > tolerance
            if constraint.sense == "="
            else lhs > constraint.rhs + tolerance
            if constraint.sense == "<="
            else lhs < constraint.rhs - tolerance
        )
        if violated:
            raise RuntimeError(
                f"Solução viola a restrição {constraint.name}: "
                f"lado esquerdo={lhs}, sentido={constraint.sense}, "
                f"lado direito={constraint.rhs}."
            )


def _extract_solution(
    solution: list[float],
    variables: dict[str, dict[tuple[int, ...], int]],
    initial_stacks: list[list[ContainerId]],
    container_order: tuple[ContainerId, ...],
    capacity: int,
    horizon: int,
) -> tuple[tuple[MIPMove, ...], tuple[tuple[ContainerId, ...], ...], int]:
    replay = BlockRelocationProblem(initial_stacks, container_order, capacity)
    moves: list[MIPMove] = []
    retrieval_periods: dict[ContainerId, int] = {}
    for t in range(horizon):
        moved = [
            c
            for c in range(len(container_order))
            if solution[variables["M"][c, t]] > 0.5
        ]
        if not moved:
            continue
        if len(moved) != 1:
            raise RuntimeError(f"Solução inválida: múltiplos M[c,t] no período {t + 1}.")
        c = moved[0]
        source_stacks = [
            s
            for s in range(len(initial_stacks))
            if solution[variables["Rc"][c, s, t]] > 0.5
        ]
        if len(source_stacks) != 1:
            raise RuntimeError(
                f"Solução inválida: contêiner {container_order[c]!r} "
                f"tem {len(source_stacks)} pilhas de origem no período {t + 1}."
            )
        source = source_stacks[0]
        if solution[variables["T"][c, t]] > 0.5:
            replay.retrieve(source)
            retrieval_periods[container_order[c]] = t + 1
            moves.append(
                MIPMove(t + 1, "retrieve", container_order[c], source)
            )
        else:
            destinations = [
                s
                for s in range(len(initial_stacks))
                if solution[variables["Pc"][c, s, t]] > 0.5
            ]
            if len(destinations) != 1:
                raise RuntimeError(
                    f"Solução inválida: contêiner {container_order[c]!r} "
                    f"tem {len(destinations)} pilhas de destino no período {t + 1}."
                )
            destination = destinations[0]
            replay.relocate(source, destination)
            moves.append(
                MIPMove(
                    t + 1,
                    "relocate",
                    container_order[c],
                    source,
                    destination,
                )
            )
        replay.validate()

    if not replay.is_complete:
        raise RuntimeError("O solver retornou um incumbente que não retira todos os contêineres.")
    if not replay.validate():
        raise RuntimeError("A sequência extraída não é uma solução BRP válida.")
    objective_period = retrieval_periods[container_order[-1]]
    return (
        tuple(moves),
        tuple(tuple(stack) for stack in replay.stacks),
        objective_period,
    )


def solve_brp3(
    problem: BlockRelocationProblem,
    *,
    horizon: int | None = None,
    backend: Backend = "auto",
    time_limit: float | None = None,
    mip_gap: float | None = None,
    output: bool = False,
    include_safety_constraints: bool = True,
) -> BRP3Result:
    """Resolve BRP-III sem modificar ``problem``.

    ``horizon=None`` reproduz a escolha de W descrita no artigo: o número de
    turnos da solução LA-1 para o mesmo estado. Resultados com incumbente mas
    sem prova de otimalidade são identificados por ``optimal=False``.
    """
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit deve ser positivo.")
    if mip_gap is not None and mip_gap < 0:
        raise ValueError("mip_gap não pode ser negativo.")

    total_start = perf_counter()
    initial_stacks, container_order, horizon = _prepare_instance(problem, horizon)
    model, variables, _ = _build_model(
        initial_stacks,
        container_order,
        problem.capacity,
        horizon,
        include_safety_constraints=include_safety_constraints,
    )
    selected_backend = _select_backend(backend)
    if selected_backend == "gurobi":
        solution, status, objective, solver_time, gap, message = _solve_gurobi(
            model, time_limit=time_limit, mip_gap=mip_gap, output=output
        )
    else:
        solution, status, objective, solver_time, gap, message = _solve_scipy(
            model, time_limit=time_limit, mip_gap=mip_gap, output=output
        )

    moves: tuple[MIPMove, ...] = ()
    final_stacks: tuple[tuple[ContainerId, ...], ...] | None = None
    relocation_count: int | None = None
    if solution is not None:
        _validate_linear_solution(model, solution)
        moves, final_stacks, objective_period = _extract_solution(
            solution,
            variables,
            initial_stacks,
            container_order,
            problem.capacity,
            horizon,
        )
        relocation_count = sum(move.action == "relocate" for move in moves)
        if objective is not None and abs(objective - objective_period) > 1e-5:
            raise RuntimeError(
                "O objetivo do solver não corresponde ao período de retirada "
                "do último contêiner na sequência extraída."
            )

    return BRP3Result(
        status=status,
        optimal=status == "OPTIMAL",
        objective=objective,
        relocation_count=relocation_count,
        runtime_seconds=perf_counter() - total_start,
        solver_runtime_seconds=solver_time,
        mip_gap=gap,
        backend=selected_backend,
        moves=moves,
        final_stacks=final_stacks,
        message=message,
    )
