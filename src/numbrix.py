# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 06:
# 95555 Diogo Venâncio
# 95675 Sofia Morgado
import copy
import sys

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

    def __init__(self, init_matrix):
        self.n = len(init_matrix)
        self.max_value = self.n * self.n
        self.matrix = init_matrix
        self.available = [i for i in range(1, self.max_value + 1)]
        self.inserted = self.build_matrix_structs()

    def build_matrix_structs(self):
        """Builds a list of available values and a dictionary with the values already there + their coordinates."""

        result = {}

        # Goes over all positions in the matrix and checks if they are already filled and stores their coordinates
        for i in range(self.n):
            for j in range(self.n):
                val = self.get_number(i, j)
                if val != 0:
                    self.available.remove(val)
                    result[val] = (i, j)

        return result

    def get_board(self):
        """Returns a matrix representation"""
        return self.matrix

    def get_number(self, row, col):
        """Returns number in requested position."""
        return self.matrix[row][col]

    def set_number(self, row, col, val):
        """Changes number in row and col to the input val."""
        self.matrix[row][col] = val

    def adjacent_vertical_numbers(self, row, col):
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

    def adjacent_horizontal_numbers(self, row, col):
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

    @staticmethod
    def parse_instance(filename):
        """Reads input file and returns a valid instance of the board class."""
        with open(filename, encoding="utf-8") as f:

            # Gets column and row sizes
            n = int(f.readline())

            # Puts rest of the lines in a matrix to represents the board
            return [[int(word) for word in f.readline().split() if word.isdigit()] for _ in range(n)]

    def get_result(self):
        """Outputs a string representation of this board"""
        return "\n".join(["\t".join([str(self.get_number(i, j)) for j in range(self.n)]) for i in range(self.n)]) + "\n"


class Numbrix(Problem):

    def __init__(self, board):
        """Specifies initial state of the board."""
        super().__init__(initial=NumbrixState(board))
        self.board = board

    def get_position_valid_values(self, state, row, col):
        """Returns a list of integers which represent the values that can be put in the input coordinates."""

        def is_valid(val1, up1, down1, left1, right1):
            if val1 != up1 and val1 != down1 and val1 != left1 and val1 != right1:
                if up1 is not None and (val1 == up1 + 1 or val1 == up1 - 1):
                    return True
                if down1 is not None and (val1 == down1 + 1 or val1 == down1 - 1):
                    return True
                if left1 is not None and (val1 == left1 + 1 or val1 == left1 - 1):
                    return True
                if right1 is not None and (val1 == right1 + 1 or val1 == right1 - 1):
                    return True
                return False

        result = []

        # Gets restriction values adjacent to the currently being evaluated position
        up, down = state.board.adjacent_vertical_numbers(row, col)
        left, right = state.board.adjacent_horizontal_numbers(row, col)

        # Goes over all the possible values for this board and appends the value if it satisfies the
        # restrictions
        for val in state.board.get_available_board_values():
            if is_valid(val, up, down, left, right):
                result.append(val)

        return result

    def actions(self, state):
        """Returns a list of actions that can be done on the input state."""

        result = []

        # Goes over each element in the board
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

    def result(self, state, action):
        """Returns the resulting state of applying the input action (result from self.action(state)) to
        the current state."""
        new_state = copy.deepcopy(state)
        new_state.board.set_number(*action)
        new_state.board.available.remove(action[2])
        return NumbrixState(new_state.board)

    def goal_test(self, state):
        """Checks if we have a valid solution of this game-"""
        return len(state.board.available) == 0

    def h(self, node):
        """Heuristic function used in A*"""
        # TODO: ideas
        #       -> Ver se o valor que está ao meu lado é +/- 1 que eu e dar mais pontos
        #       -> Dar mais pontos à medida que vai formando um caminho (Muito lento)
        #       -> Somar linhas/colunas ??? (ideia do stor)
        return 1


if __name__ == "__main__":

    # Creates matrix that represents game board from input file
    board = Board(Board.parse_instance(sys.argv[1]))

    # Initializes Numbrix problem
    numbrix = Numbrix(board)

    # Applies our search algorithm to find the correct solution
    result = astar_search(numbrix).state.board.get_result()

    # Shows result in stdin
    print(result, end="")
