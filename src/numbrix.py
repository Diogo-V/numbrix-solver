# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 06:
# 95555 Diogo Venâncio
# 95675 Sofia Morgado

import sys
from typing import Tuple

from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search


class Action:
    """Encapsulates action representation."""

    def __init__(self, row, column, number):
        self.row = row
        self.column = column
        self.number = number

    def get_row(self):
        return self.row

    def get_column(self):
        return self.column

    def get_number(self):
        return self.number

    def to_tuple(self):
        return self.row, self.column, self.number


class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id
        

class Board:
    """Numbrix board representation."""

    def __init__(self, init_matrix: list[list[int]]):
        self.n = len(init_matrix)
        self.max_value = self.n * self.n
        self.matrix = init_matrix
    
    def get_number(self, row: int, col: int) -> int:
        """Returns number in requested position."""
        return self.matrix[row][col]

    def set_number(self, row: int, col: int, val: int) -> None:
        """Changes number in row and col to the input val."""
        self.matrix[row][col] = val

    def adjacent_vertical_numbers(self, row: int, col: int) -> tuple:
        """Returns values in upper and lower positions, respectively."""
        result = ()
        try:
            result += (self.matrix[row + 1][col], )
        except IndexError:
            result += (None, )
        try:
            result += (self.matrix[row - 1][col], )
        except IndexError:
            result += (None, )
        return result

    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple:
        """ Returns values to the left and to right, respectively."""
        result = ()
        try:
            result += (self.matrix[row][col - 1], )
        except IndexError:
            result += (None, )
        try:
            result += (self.matrix[row][col + 1], )
        except IndexError:
            result += (None, )
        return result

    def get_available_board_values(self) -> list[int]:
        """Returns a list with the currently not being used possible values for this board."""
        return [val for val in range(1, self.max_value + 1) if val not in [j for sub in self.matrix for j in sub]]
    
    @staticmethod    
    def parse_instance(filename: str) -> list[list[int]]:
        """Reads input file and returns a valid instance of the board class."""
        with open(filename, encoding="utf-8") as f:

            # Gets column and row sizes
            n = int(f.readline())

            # Puts rest of the lines in a matrix to represents the board
            return [[int(word) for word in f.readline().split() if word.isdigit()] for _ in range(n)]

    def get_result(self) -> str:
        """Outputs a string representation of this board"""
        return "\n".join(["\t".join([str(self.get_number(i, j)) for j in range(self.n)]) for i in range(self.n)]) + "\n"


class Numbrix(Problem):

    def __init__(self, board: Board):
        """Specifies initial state of the board."""
        super().__init__(initial=NumbrixState(board))
        self.board = board
        self.available = self.board.get_available_board_values()

    def actions(self, state: NumbrixState) -> list[Action]:
        """Returns a list of actions that can be done on the input state."""

        result = []

        # Goes over each element in the
        for i in range(self.board.n):
            for j in range(self.board.n):

                # Checks if this position is not already filled
                if self.board.get_number(i, j) == 0:

                    # Gets restriction values adjacent to the currently being evaluated position
                    up, down = self.board.adjacent_vertical_numbers(i, j)
                    left, right = self.board.adjacent_horizontal_numbers(i, j)

                    # Goes over all the possible values for this board and appends the value if it satisfies the
                    # restrictions
                    for val in self.available:
                        if val != up and val != down and val != left and val != right:
                            result.append(Action(i, j, val))

        return result

    def result(self, state: NumbrixState, action: Action) -> NumbrixState:
        """Returns the resulting state of applying the input action (result from self.action(state)) to
        the current state."""

        # Removes value from list of available values for this board
        self.available.remove(action[2])

        # Changes board value to the requested action and returns it
        self.board.set_number(*action.to_tuple())
        return NumbrixState(self.board)

    def goal_test(self, state: NumbrixState) -> bool:
        """Checks if we have a valid solution of this game-"""

        # Stores the start of this board (number one)
        row, col = 0, 0

        # Finds row and column where the solution starts (finds value=1)
        for i in range(self.board.n):
            for j in range(self.board.n):
                if self.board.get_number(i, j) == 1:
                    row, col = i, j

        # Tries to go through the solution's path and check whether it is valid or not
        val = 1
        while val < self.board.max_value:

            # Gets adjacent values to discover where is the next value located
            up, down = self.board.adjacent_vertical_numbers(row, col)
            left, right = self.board.adjacent_horizontal_numbers(row, col)

            # Tries to see where is the next position
            if up == self.board.get_number(row, col) + 1:
                row -= 1
            elif down == self.board.get_number(row, col) + 1:
                row += 1
            elif left == self.board.get_number(row, col) + 1:
                col -= 1
            elif right == self.board.get_number(row, col) + 1:
                col += 1
            else:
                return False  # In this case, we were not able to find a suitable path and so, this is not a solution

            val += 1

        return True

    def h(self, node: Node):
        """Heuristic function used in A*"""
        # TODO
        pass
    

if __name__ == "__main__":

    # Reads input file
    if len(sys.argv) == 2:

        # Creates matrix that represents game board
        board = Board(Board.parse_instance(sys.argv[1]))

        # Initializes Numbrix problem
        numbrix = Numbrix(board)

        print("A*: \n", astar_search(numbrix))

        # Prints solution in the stdout
        print(board.get_result())

    else:
        print("Invalid number of arguments. Only needs a path to be passed!")
