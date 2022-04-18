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
        self.inserted = self.build_matrix_structs()

    def build_matrix_structs(self):
        """Builds a list of available values and a dictionary with the values already there + their coordinates."""

        result = {}

        # Goes over all positions in the matrix and checks if they are already filled and stores their coordinates
        for i in range(self.n):
            for j in range(self.n):
                val = self.get_number(i, j)
                if val != 0:
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

    def actions(self, state):
        """Returns a list of actions that can be done on the input state."""

        result = []

        # Iterates over all the inserted items and returns the possible action for each one of them
        for key, (row, col) in state.board.inserted.items():

            # Gets restriction values adjacent to the currently being evaluated position
            up, down = state.board.adjacent_vertical_numbers(row, col)
            left, right = state.board.adjacent_horizontal_numbers(row, col)

            # Gets available adjacent values
            available = [i for i in [key + 1, key - 1] if i not in state.board.inserted]

            # Iterates over the possible adjacent values and appends possible actions to the result list
            for val in available:

                # We need to check if our adjacency is an empty position before appending the action
                if up == 0:
                    result.append((row - 1, col, val))
                if down == 0:
                    result.append((row + 1, col, val))
                if left == 0:
                    result.append((row, col - 1, val))
                if right == 0:
                    result.append((row, col + 1, val))

        return result

    def result(self, state, action):
        """Returns the resulting state of applying the input action (result from self.action(state)) to
        the current state."""
        new_state = copy.deepcopy(state)
        new_state.board.set_number(*action)
        return NumbrixState(new_state.board)

    def goal_test(self, state):
        """Checks if we have a valid solution of this game-"""
        return len(state.board.inserted) == state.board.max_value

    def h(self, node):
        """Heuristic function used in A*"""
        # TODO: ideas
        #       -> Ver se o valor que está ao meu lado é +/- 1 que eu e dar mais pontos
        #       -> Dar mais pontos à medida que vai formando um caminho (Muito lento)
        #       -> Somar linhas/colunas ??? (ideia do stor)
        #       -> Modelo epidemiológico?
        #       -> Calculo de distancias entre valores?
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
