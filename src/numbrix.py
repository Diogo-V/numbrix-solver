# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 06:
# 95555 Diogo Venâncio
# 95675 Sofia Morgado
import copy
import sys
from typing import Tuple

from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search


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
        self.available = self.get_available_board_values()

    def get_board(self) -> list[list[int]]:
        """Returns a matrix representation"""
        return self.matrix
    
    def get_number(self, row: int, col: int) -> int:
        """Returns number in requested position."""
        return self.matrix[row][col]

    def set_number(self, row: int, col: int, val: int) -> None:
        """Changes number in row and col to the input val."""
        self.matrix[row][col] = val

    def adjacent_vertical_numbers(self, row: int, col: int) -> tuple:
        """Returns values in upper and lower positions, respectively."""
        result = ()
        if row == 0:
            result += (None, )
        else:
            result += (self.matrix[row - 1][col], )
        if row == self.n - 1:
            result += (None, )
        else:
            result += (self.matrix[row + 1][col],)
        return result

    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple:
        """ Returns values to the left and to right, respectively."""
        result = ()
        if col == 0:
            result += (None, )
        else:
            result += (self.matrix[row][col - 1], )
        if col == self.n - 1:
            result += (None, )
        else:
            result += (self.matrix[row][col + 1], )
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

    def get_position_valid_values(self, state: NumbrixState, row: int, col: int) -> list[int]:
        """Returns a list of integers which represent the values that can be put in the input coordinates."""

        result = []

        # Gets restriction values adjacent to the currently being evaluated position
        up, down = state.board.adjacent_vertical_numbers(row, col)
        left, right = state.board.adjacent_horizontal_numbers(row, col)

        # Goes over all the possible values for this board and appends the value if it satisfies the
        # restrictions
        for val in state.board.get_available_board_values():
            if val != up and val != down and val != left and val != right:
                result.append(val)

        return result

    def actions(self, state: NumbrixState) -> list[tuple[int, int, int]]:
        """Returns a list of actions that can be done on the input state."""

        result = []

        # Goes over each element in the
        for i in range(state.board.n):
            for j in range(state.board.n):

                # Checks if this position is not already filled
                if state.board.get_number(i, j) == 0:

                    # Get values that can be put in this position
                    values = self.get_position_valid_values(state, i, j)

                    # Appends possible actions for this position
                    for val in values:
                        result.append((i, j, val))

        return result

    def result(self, state: NumbrixState, action: tuple[int, int, int]) -> NumbrixState:
        """Returns the resulting state of applying the input action (result from self.action(state)) to
        the current state."""

        # Removes value from list of available values for this board
        # self.available.remove(action[2])  TODO: decide what to do with this

        # Changes board value to the requested action and returns it
        # new_board = copy.deepcopy(self.board)  TODO: make sure it is not this
        # new_board.set_number(*action)
        new_state = copy.deepcopy(state)
        new_state.board.set_number(*action)
        new_state.board.available.remove(action[2])
        return NumbrixState(new_state.board)

    def goal_test(self, state: NumbrixState) -> bool:
        """Checks if we have a valid solution of this game-"""

        # Stores the start of this board (number one)
        row, col = 0, 0

        # Finds row and column where the solution starts (finds value=1)
        for i in range(state.board.n):
            for j in range(state.board.n):
                if state.board.get_number(i, j) == 1:
                    row, col = i, j

        # Tries to go through the solution's path and check whether it is valid or not
        val = 1
        while val < state.board.max_value:

            # Gets adjacent values to discover where is the next value located
            up, down = state.board.adjacent_vertical_numbers(row, col)
            left, right = state.board.adjacent_horizontal_numbers(row, col)

            # Tries to see where is the next position
            if up == state.board.get_number(row, col) + 1:
                row -= 1
            elif down == state.board.get_number(row, col) + 1:
                row += 1
            elif left == state.board.get_number(row, col) + 1:
                col -= 1
            elif right == state.board.get_number(row, col) + 1:
                col += 1
            else:
                return False  # In this case, we were not able to find a suitable path and so, this is not a solution

            val += 1

        return True

    def h(self, node: Node):
        """Heuristic function used in A*"""
        # TODO: ideas
        #       -> Ver se o valor que está ao meu lado é +/- 1 que eu e dar mais pontos
        #       -> Dar mais pontos à medida que vai formando um caminho
        #       -> Somar linhas/colunas ??? (ideia do stor)

        # All nodes get a +1 value from their heuristic function
        total = 1

        # We initialize our loop condition a true to be able to find a path
        found_path = True

        # This value is going to be used to "reward" a good path. Each time we find a path, this value is doubled. This
        # allows us to be able to "reward" longer paths
        total_to_be_added = 2

        # Holds list of visited values (used to not repeat already visited positions)
        visited = []

        # If this node does not have an action applied to it, then we don't need to find a path
        if node.action is None:
            return total

        # Gets action that is going to be taken on this node and unpacks it to get the adjacent nodes
        row, col, val = node.action

        while found_path:

            # Gets adjacent values
            up, down = node.state.board.adjacent_vertical_numbers(row, col)
            left, right = node.state.board.adjacent_horizontal_numbers(row, col)

            # Now, we try to form a path from this node by evaluating all adjacent nodes and see if they form a path too
            # and if so, we update the current value and double the next score if we can find even more nodes that form
            # a path
            if up is not None and up != 0 and up not in visited:
                if val == up + 1 or val == up - 1:
                    total += total_to_be_added
                    total_to_be_added *= 2
                    row -= 1
                    continue

            if down is not None and down != 0 and down not in visited:
                if val == down + 1 or val == down - 1:
                    total += total_to_be_added
                    total_to_be_added *= 2
                    row += 1
                    continue

            if left is not None and left != 0 and left not in visited:
                if val == left + 1 or val == left - 1:
                    total += total_to_be_added
                    total_to_be_added *= 2
                    col -= 1
                    continue

            if right is not None and right != 0 and right not in visited:
                if val == right + 1 or val == right - 1:
                    total += total_to_be_added
                    total_to_be_added *= 2
                    col += 1
                    continue

            # In case no valid path is found, then we stop this loop and return the total accumulated heuristic value
            found_path = False

        return total
    

if __name__ == "__main__":

    # Reads input file
    if len(sys.argv) == 2:

        # Creates matrix that represents game board
        board = Board(Board.parse_instance(sys.argv[1]))

        # Initializes Numbrix problem
        numbrix = Numbrix(board)

        print("A*: \n", astar_search(numbrix, display=True))

        # Prints solution in the stdout
        print(board.get_result())

    else:
        print("Invalid number of arguments. Only needs a path to be passed!")
