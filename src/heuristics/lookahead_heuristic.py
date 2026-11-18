"""Heurística Extended Look-Ahead LA-N de Petering e Hussein (2013)."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Literal

from src.problem import BlockRelocationProblem, ContainerId


@dataclass(frozen=True)
class LookaheadMove:
    """Um passo da solução; índices de pilha começam em zero."""

    action: Literal["relocate", "retrieve"]
    container: ContainerId
    source: int
    destination: int | None = None
    cleaning: bool = False


@dataclass(frozen=True)
class LookaheadResult:
    """Resultado da execução de LA-N sobre uma instância."""

    lookahead: int
    moves: tuple[LookaheadMove, ...]
    final_stacks: tuple[tuple[ContainerId, ...], ...]

    @property
    def relocation_count(self) -> int:
        """Número de relocations, sem contar retiradas definitivas."""
        return sum(move.action == "relocate" for move in self.moves)


def solve_lookahead(
    problem: BlockRelocationProblem, n: int
) -> LookaheadResult:
    """Executa LA-N e atualiza ``problem`` até retirar todos os contêineres.

    A ordem de retirada da instância é usada como a numeração de prioridade
    dos contêineres descrita no artigo. Cada chamada registra somente os
    movimentos que ela própria executa; a instância é modificada in-place.

    Para pilhas vazias, ``Low(s)`` é tratado como infinito: colocá-las é uma
    escolha boa (não há contêineres que possam voltar a bloquear o movido).
    Se várias pilhas vazias tiverem o mesmo valor, escolhe-se a de menor
    índice, pois o artigo não especifica desempate entre destinos equivalentes.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("O parâmetro N deve ser um inteiro positivo.")

    priority = {
        container: index for index, container in enumerate(problem.retrieval_order)
    }
    container_stack = {
        container: stack_index
        for stack_index, stack in enumerate(problem.stacks)
        for container in stack
    }
    retrieved = set(problem.retrieved)
    moves: list[LookaheadMove] = []

    def stack_containing(container: ContainerId) -> int:
        try:
            return container_stack[container]
        except KeyError as error:
            raise RuntimeError(
                f"Contêiner {container!r} não encontrado no problema."
            ) from error

    def low_rank(stack_index: int) -> float | int:
        lowest = problem.minimum_container(stack_index)
        return inf if lowest is None else priority[lowest]

    while not problem.is_complete:
        target = problem.next_container
        if target is None:
            raise RuntimeError("Estado inconsistente: falta o próximo contêiner-alvo.")
        target_stack = stack_containing(target)

        if problem.top(target_stack) == target:
            problem.retrieve(target_stack)
            moves.append(LookaheadMove("retrieve", target, target_stack))
            retrieved.add(target)
            del container_stack[target]
            continue

        remaining = [
            container
            for container in problem.retrieval_order
            if container not in retrieved
        ]
        n0 = min(n, len(remaining))

        # Reduz N0 até haver uma pilha fora de Stacks[N0] com espaço.
        while n0 > 0:
            stacks_n0 = {
                stack_containing(container) for container in remaining[:n0]
            }
            outside = set(range(problem.stack_count)) - stacks_n0
            if len(stacks_n0) == problem.stack_count or all(
                problem.height(stack_index) == problem.capacity
                for stack_index in outside
            ):
                n0 -= 1
                continue
            break

        if n0 == 0:
            raise ValueError(
                "LA-N não pode prosseguir: não há pilha de destino com espaço "
                "fora da pilha do contêiner-alvo. O artigo não define o caso "
                "sem relocação factível sob a capacidade atual."
            )

        stacks_n0 = {stack_containing(container) for container in remaining[:n0]}
        candidate_tops = sorted(
            (
                problem.top(stack_index)
                for stack_index in stacks_n0
                if problem.top(stack_index) is not None
            ),
            key=priority.__getitem__,
            reverse=True,
        )

        selected: ContainerId | None = None
        selected_source: int | None = None
        for candidate in candidate_tops:
            source = stack_containing(candidate)

            # Step 4/6: se o topo é o do alvo, sua relocação é obrigatória.
            if source == target_stack:
                selected = candidate
                selected_source = source
                break

            # Step 5: só antecipar um cleaning move se ele for bom.
            destinations = [
                stack_index
                for stack_index in range(problem.stack_count)
                if stack_index != source
                and problem.height(stack_index) < problem.capacity
                and low_rank(stack_index) > priority[candidate]
            ]
            if problem.minimum_container(source) == candidate or not destinations:
                continue

            selected = candidate
            selected_source = source
            break

        if selected is None or selected_source is None:
            raise RuntimeError(
                "LA-N não encontrou o topo da pilha-alvo entre os candidatos; "
                "a configuração viola uma invariante do algoritmo."
            )

        # Step 6: usa o melhor destino bom; se não houver, adia a próxima
        # relocação escolhendo a pilha disponível com Low(s) máximo.
        good_destinations = [
            stack_index
            for stack_index in range(problem.stack_count)
            if stack_index != selected_source
            and problem.height(stack_index) < problem.capacity
            and low_rank(stack_index) > priority[selected]
        ]
        if good_destinations:
            destination = min(good_destinations, key=low_rank)
        else:
            available_destinations = [
                stack_index
                for stack_index in range(problem.stack_count)
                if stack_index != selected_source
                and problem.height(stack_index) < problem.capacity
            ]
            if not available_destinations:
                raise ValueError(
                    "Não há outra pilha com espaço para relocar o contêiner "
                    f"{selected!r} sem exceder a capacidade."
                )
            destination = max(available_destinations, key=low_rank)

        problem.relocate(selected_source, destination)
        container_stack[selected] = destination
        moves.append(
            LookaheadMove(
                action="relocate",
                container=selected,
                source=selected_source,
                destination=destination,
                cleaning=selected_source != target_stack,
            )
        )

    problem.validate()
    return LookaheadResult(
        lookahead=n,
        moves=tuple(moves),
        final_stacks=tuple(tuple(stack) for stack in problem.stacks),
    )
