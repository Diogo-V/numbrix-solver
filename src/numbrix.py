# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 06:
# 95555 Diogo Venâncio
# 95675 Sofia Morgado
import copy
import heapq
import sys
import bisect
import collections

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
        self.deque = collections.deque()
        self.clusters = {}
        self.inserted = {}
        self.pointer = None
        self.reached_end = False
        self.build_matrix_structs()

    def build_matrix_structs(self):
        """Builds a dictionary with the values already in the board + their coordinates and also setups the frontier."""

        # Goes over all positions in the matrix and checks if they are already filled and stores their coordinates
        for i in range(self.n):
            for j in range(self.n):
                val = self.get_number(i, j)
                if val != 0:
                    self.inserted[val] = (i, j)

        # Builds frontiers after knowing which nodes have been inserted
        for val in self.inserted.keys():
            Board.insert_deque(self, val)

        if self.deque[0] in self.clusters:
            opposite, _ = self.clusters[self.deque[0]]
            self.pointer = (self.inserted[opposite][0], self.inserted[opposite][1], opposite)

    @staticmethod
    def get_possible_values(board, row, col, inserted):
        """Returns possible values for input coordinate. Returns empty list if none are found."""

        def check(border):
            result1 = []
            if border is not None and border != 0:
                if border - 1 not in inserted and 0 < border - 1 <= board.max_value and border - 1 not in result:
                    result1.append(border - 1)
                if border + 1 not in inserted and 0 < border + 1 <= board.max_value and border + 1 not in result:
                    result1.append(border + 1)
            return result1

        result = []

        # Gets restriction values adjacent to the currently being evaluated position
        up, down = board.adjacent_vertical_numbers(row, col)
        left, right = board.adjacent_horizontal_numbers(row, col)

        # Appends possible values
        result.extend(check(up))
        result.extend(check(down))
        result.extend(check(left))
        result.extend(check(right))

        return result

    @staticmethod
    def insert_deque(board, value):
        """Inserts a value in our deque structure."""

        def can_merge_cluster():
            row, col = board.inserted[value]
            up, down = board.adjacent_vertical_numbers(row, col)
            left, right = board.adjacent_horizontal_numbers(row, col)
            return len({up, down, left, right} & {value + 1}) > 0

        def is_adjacent_left():
            row, col = board.inserted[value]
            up, down = board.adjacent_vertical_numbers(row, col)
            left, right = board.adjacent_horizontal_numbers(row, col)
            return len({up, down, left, right} & {value - 1}) > 0

        # Inserts value in our deque structure
        i = bisect.bisect_left(board.deque, value)
        board.deque.insert(i, value)

        # Now we need to check cluster conditions and update it if the new value becomes a 'lance' node
        val_left = -1
        val_right = -1

        if i - 1 >= 0:
            if board.deque[i - 1] == value - 1 and is_adjacent_left():
                val_left = board.deque[i - 1]
                opposite_lance, degree = board.clusters[val_left]
                board.clusters[value] = (opposite_lance, degree + 1)
                board.clusters[opposite_lance] = (value, degree + 1)
                if degree > 1:  # Takes care of a cluster with a single value (does not allow deletion of it)
                    board.clusters.pop(val_left)

        if i + 1 < len(board.deque):
            if board.deque[i + 1] == value + 1:
                val_right = board.deque[i + 1]
                opposite_lance, degree = board.clusters[val_right]
                if val_left != -1:  # We need to check if we have joined two clusters
                    if can_merge_cluster():
                        new_degree = degree + board.clusters[value][1]
                        board.clusters[opposite_lance] = (board.clusters[value][0], new_degree)
                        board.clusters[board.clusters[value][0]] = (opposite_lance, new_degree)
                        board.clusters.pop(value)
                        if degree != 1:
                            board.clusters.pop(value + 1)
                        return opposite_lance
                else:
                    board.clusters[value] = (opposite_lance, degree + 1)
                    board.clusters[opposite_lance] = (value, degree + 1)
                    if degree > 1:  # Takes care of a cluster with a single value (does not allow deletion of it)
                        board.clusters.pop(val_right)

        # This is the first value being put in the deque structure
        if val_left == -1 and val_right == -1:
            board.clusters[value] = (value, 1)

    @staticmethod
    def get_deque_index(board, value):
        return bisect.bisect_left(board.deque, value)

    @staticmethod
    def calc_distance(board, action, target):
        row, col, val = action
        upper = abs(val - target)
        target_row, target_col = board.inserted[target]
        lower = abs(row - target_row) + abs(col - target_col)
        result = upper / lower
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

        # Unpacks action that was taken previously
        row, col, val = state.board.pointer

        # We also need to update our own adjacent frontier values
        up, down = state.board.adjacent_vertical_numbers(row, col)
        left, right = state.board.adjacent_horizontal_numbers(row, col)

        # Frontier of the currently being evaluated position
        frontier = {}

        # Possible values
        pos = []
        if state.board.reached_end:
            values_pos = [val - 1]
        else:
            values_pos = [val + 1]
        for value in values_pos:
            if value not in state.board.inserted and 0 < value <= state.board.max_value:
                pos.append(value)

        # No more possibilities
        if not pos:
            return []

        # Updates frontier
        if up == 0:
            frontier[(row - 1, col)] = pos
        if down == 0:
            frontier[(row + 1, col)] = pos
        if left == 0:
            frontier[(row, col - 1)] = pos
        if right == 0:
            frontier[(row, col + 1)] = pos

        if len(state.board.clusters) == 2:  # In this case, we have already filled all the cracks between the clusters

            return [(row, col, val) for (row, col), values in frontier.items() for val in values]

        else:  # We still haven't gotten to the highest element

            # Gets element that we are trying to get to
            my_index = Board.get_deque_index(state.board, val)
            next_value = state.board.deque[my_index + 1]

            tmp = []
            heapq.heapify(tmp)

            # Builds heap to prioritize lower distance values
            for (row, col), values in frontier.items():
                for possibility in values:
                    dist = Board.calc_distance(state.board, (row, col, possibility), next_value)
                    if dist >= 1:
                        heapq.heappush(tmp, (dist, (row, col, possibility)))

            result = []
            for val in tmp:
                result.append(val[1])
            return result

    def result(self, state, action):
        """Returns the resulting state of applying the input action (result from self.action(state)) to
        the current state."""

        def is_valid_adjacent(board):
            row, col = board.inserted[action[2]]
            up, down = board.adjacent_vertical_numbers(row, col)
            left, right = board.adjacent_horizontal_numbers(row, col)
            return len({up, down, left, right} & {action[2] + 1}) > 0

        new_state = copy.deepcopy(state)
        new_state.board.set_number(*action)
        new_state.board.inserted[action[2]] = (action[0], action[1])
        new_max = new_state.board.insert_deque(new_state.board, action[2])
        new_state.board.pointer = action
        if action[2] == new_state.board.max_value or new_max == new_state.board.max_value:
            new_state.board.reached_end = True
            new_state.board.pointer = (*new_state.board.inserted[new_state.board.deque[0]], new_state.board.deque[0])
            return NumbrixState(new_state.board)
        if not new_state.board.reached_end:
            i = Board.get_deque_index(new_state.board, action[2] + 1)
            if i + 1 < len(new_state.board.deque) or new_max is not None:
                if new_state.board.deque[i] == action[2] + 1 and is_valid_adjacent(new_state.board):
                    new_state.board.pointer = (*new_state.board.inserted[new_max], new_max)
        return NumbrixState(new_state.board)

    def goal_test(self, state):
        """Checks if we have a valid solution of this game-"""
        return len(state.board.inserted) == state.board.max_value

    def h(self, node):
        """Heuristic function used in A*"""
        return 1


if __name__ == "__main__":

    # Creates matrix that represents game board from input file
    instance = Board(Board.parse_instance(sys.argv[1]))

    # Initializes Numbrix problem
    numbrix = Numbrix(instance)

    # Applies our search algorithm to find the correct solution
    solved = depth_first_tree_search(numbrix).state.board.get_result()

    # Shows result in stdin
    print(solved, end="")
