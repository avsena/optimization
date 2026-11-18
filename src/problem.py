"""Representação e regras básicas do Block Relocation Problem (BRP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence


ContainerId = Hashable


@dataclass(frozen=True)
class Relocation:
    """Registro de uma relocação, com pilhas indexadas a partir de zero."""

    container: ContainerId
    source: int
    destination: int


class BlockRelocationProblem:
    """Estado de uma instância do BRP.

    Cada pilha é armazenada do fundo para o topo. A ordem de retirada define
    a prioridade dos contêineres: o primeiro da sequência deve ser retirado
    antes dos demais, independentemente do identificador usado.
    """

    def __init__(
        self,
        stacks: Sequence[Iterable[ContainerId]],
        retrieval_order: Sequence[ContainerId],
        capacity: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("A capacidade das pilhas deve ser positiva.")
        if not stacks:
            raise ValueError("A instância deve conter pelo menos uma pilha.")

        self.capacity = capacity
        self.stacks = [list(stack) for stack in stacks]
        self.retrieval_order = tuple(retrieval_order)

        if any(len(stack) > capacity for stack in self.stacks):
            raise ValueError("Uma pilha inicial excede a capacidade.")
        if len(set(self.retrieval_order)) != len(self.retrieval_order):
            raise ValueError("A ordem de retirada contém identificadores duplicados.")

        self._priority = {
            container: index for index, container in enumerate(self.retrieval_order)
        }
        initial_containers = [
            container for stack in self.stacks for container in stack
        ]
        if len(set(initial_containers)) != len(initial_containers):
            raise ValueError("Um contêiner aparece em mais de uma posição.")
        if set(initial_containers) != set(self.retrieval_order):
            raise ValueError(
                "A ordem de retirada deve conter exatamente os contêineres iniciais."
            )

        self._next_retrieval = 0
        self.relocations: list[Relocation] = []
        self.retrieved: list[ContainerId] = []

    @property
    def stack_count(self) -> int:
        """Número de pilhas da instância."""
        return len(self.stacks)

    @property
    def next_container(self) -> ContainerId | None:
        """Próximo contêiner que deve ser retirado, ou ``None`` se concluído."""
        if self.is_complete:
            return None
        return self.retrieval_order[self._next_retrieval]

    @property
    def is_complete(self) -> bool:
        """Indica se todos os contêineres foram retirados."""
        return self._next_retrieval == len(self.retrieval_order)

    @property
    def relocation_count(self) -> int:
        """Número total de relocations realizadas até o momento."""
        return len(self.relocations)

    def _check_stack_index(self, stack_index: int) -> None:
        if not isinstance(stack_index, int) or not 0 <= stack_index < self.stack_count:
            raise IndexError(f"Índice de pilha inválido: {stack_index}.")

    def top(self, stack_index: int) -> ContainerId | None:
        """Retorna o contêiner no topo, ou ``None`` para uma pilha vazia."""
        self._check_stack_index(stack_index)
        stack = self.stacks[stack_index]
        return stack[-1] if stack else None

    def height(self, stack_index: int) -> int:
        """Retorna a altura atual de uma pilha."""
        self._check_stack_index(stack_index)
        return len(self.stacks[stack_index])

    def minimum_container(self, stack_index: int) -> ContainerId | None:
        """Retorna o contêiner de maior prioridade de retirada na pilha.

        ``None`` é retornado para uma pilha vazia. A prioridade é determinada
        pela posição na ordem de retirada, não pelo valor do identificador.
        """
        self._check_stack_index(stack_index)
        stack = self.stacks[stack_index]
        if not stack:
            return None
        return min(stack, key=self._priority.__getitem__)

    def relocate(self, source: int, destination: int) -> ContainerId:
        """Move o contêiner do topo entre pilhas sem exceder sua capacidade."""
        self._check_stack_index(source)
        self._check_stack_index(destination)
        if source == destination:
            raise ValueError("A pilha de origem e destino devem ser diferentes.")
        if not self.stacks[source]:
            raise ValueError("Não é possível mover de uma pilha vazia.")
        if len(self.stacks[destination]) >= self.capacity:
            raise ValueError("A pilha de destino está cheia.")

        container = self.stacks[source].pop()
        self.stacks[destination].append(container)
        self.relocations.append(Relocation(container, source, destination))
        return container

    def retrieve(self, stack_index: int) -> ContainerId:
        """Retira o próximo contêiner requerido, respeitando a regra LIFO."""
        self._check_stack_index(stack_index)
        if self.is_complete:
            raise ValueError("Todos os contêineres já foram retirados.")
        if not self.stacks[stack_index]:
            raise ValueError("Não é possível retirar de uma pilha vazia.")

        expected = self.retrieval_order[self._next_retrieval]
        if self.stacks[stack_index][-1] != expected:
            raise ValueError(
                f"O próximo contêiner é {expected!r}; ele deve estar no topo."
            )

        container = self.stacks[stack_index].pop()
        self.retrieved.append(container)
        self._next_retrieval += 1
        return container

    def validate(self) -> bool:
        """Verifica capacidade, unicidade e conservação dos contêineres."""
        if any(len(stack) > self.capacity for stack in self.stacks):
            raise AssertionError("Uma pilha excedeu sua capacidade.")
        present = [container for stack in self.stacks for container in stack]
        all_accounted_for = present + self.retrieved
        if len(all_accounted_for) != len(set(all_accounted_for)):
            raise AssertionError("Há contêineres duplicados ou retirados mais de uma vez.")
        if set(all_accounted_for) != set(self.retrieval_order):
            raise AssertionError("Há contêineres perdidos ou não pertencentes à instância.")
        if tuple(self.retrieved) != self.retrieval_order[: self._next_retrieval]:
            raise AssertionError("A sequência de retiradas não respeita a ordem requerida.")
        return True


def small_instance() -> BlockRelocationProblem:
    """Cria uma instância pequena com duas relocations necessárias."""
    return BlockRelocationProblem(
        stacks=[
            [1, 4],
            [2, 3],
            [],
        ],
        retrieval_order=[1, 2, 3, 4],
        capacity=3,
    )


if __name__ == "__main__":
    problem = small_instance()

    assert problem.top(0) == 4
    assert problem.minimum_container(0) == 1
    assert problem.height(2) == 0

    assert problem.relocate(0, 2) == 4
    assert problem.retrieve(0) == 1
    assert problem.relocate(1, 0) == 3
    assert problem.retrieve(1) == 2
    assert problem.retrieve(0) == 3
    assert problem.retrieve(2) == 4

    assert problem.is_complete
    assert problem.relocation_count == 2
    assert problem.validate()
    print(
        "Instância pequena validada: "
        f"{problem.relocation_count} relocations, "
        f"retiradas={problem.retrieved}."
    )
