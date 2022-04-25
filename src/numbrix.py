# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 06:
# 95555 Diogo Venâncio
# 95675 Sofia Morgado
import copy
import sys
import bisect
import collections
import time

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
        self.cluster_manager = collections.deque()
        self.clusters = {}
        self.possibilities = {}
        self.did_clusters_merge = False
        self.inserted = {}
        self.frontier = {}
        self.previous_action = None
        self.build_matrix_structs()

    def build_matrix_structs(self):
        """Builds a dictionary with the values already in the board + their coordinates and also setups the frontier."""

        def build_frontier(row, col, board):
            if 0 <= row < board.n and 0 <= col < board.n:
                if board.matrix[row][col] == 0:
                    if (row, col) not in self.frontier:
                        possible_values = Board.get_possible_values(self, row, col, self.inserted)
                        self.frontier[(row, col)] = possible_values

        # Goes over all positions in the matrix and checks if they are already filled and stores their coordinates
        for i in range(self.n):
            for j in range(self.n):
                val = self.get_number(i, j)
                if val != 0:
                    self.inserted[val] = (i, j)

        # Builds frontiers after knowing which nodes have been inserted
        for val, (i, j) in self.inserted.items():
            build_frontier(i + 1, j, self)
            build_frontier(i - 1, j, self)
            build_frontier(i, j + 1, self)
            build_frontier(i, j - 1, self)
            Board.insert_deque(self, val)

    @staticmethod
    def add_possibility(board, value):
        """Adds/Increments the number of possibilities for a given value."""
        if value in board.possibilities:
            board.possibilities[value] += 1
        else:
            board.possibilities[value] = 1

    @staticmethod
    def delete_possibility(board, deleted_value, coordinates_remainder):
        try:
            for val in coordinates_remainder:
                if val == deleted_value:
                    board.possibilities.pop(deleted_value)
                else:
                    board.decrement_possibility(board, val)
        except KeyError:
            pass

    @staticmethod
    def decrement_possibility(board, value):
        try:
            board.possibilities[value] -= 1
        except KeyError:
            pass

    @staticmethod
    def clear_possibilities(board, row, col):
        """Clears possible values from input coordinates."""
        try:
            if (row, col) in board.frontier:
                for val in board.frontier[(row, col)]:
                    board.decrement_possibility(board, val)
        except KeyError:
            pass

    @staticmethod
    def get_possible_values(board, row, col, inserted):
        """Returns possible values for input coordinate. Returns empty list if none are found."""

        def check(border):
            result1 = []
            if border is not None and border != 0:
                if border - 1 not in inserted and 0 < border - 1 <= board.max_value and border - 1 not in result:
                    result1.append(border - 1)
                    board.add_possibility(board, border - 1)
                if border + 1 not in inserted and 0 < border + 1 <= board.max_value and border + 1 not in result:
                    result1.append(border + 1)
                    board.add_possibility(board, border + 1)
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
    def update_adjacent_frontier(board, row, col, value):
        """Updates board frontier values when 'value' is inserted in input coordinates."""

        # Gets currently open adjacent coordinates (these are the only ones that can get updated)
        up, down = board.adjacent_vertical_numbers(row, col)
        left, right = board.adjacent_horizontal_numbers(row, col)

        if up == 0:
            board.remove_frontier(board, row - 1, col, value)
        if down == 0:
            board.remove_frontier(board, row + 1, col, value)
        if left == 0:
            board.remove_frontier(board, row, col - 1, value)
        if right == 0:
            board.remove_frontier(board, row, col + 1, value)

    @staticmethod
    def remove_frontier(board, row, col, value):
        """Removes value from frontier input coordinate."""
        try:
            board.frontier[(row, col)].remove(value)
            board.decrement_possibility(board, value)
        except ValueError:
            pass

    @staticmethod
    def insert_cluster_manager(board, value):
        """Inserts a new value in our cluster manager structure."""
        board.cluster_manager.insert(bisect.bisect_left(board.cluster_manager, value), value)

    @staticmethod
    def get_smallest_cluster_size(board):
        """Returns smallest cluster size."""
        return board.cluster_manager[0]

    @staticmethod
    def remove_value_cluster_manager(board, value):
        """Removes cluster size from our cluster manager structure."""
        board.cluster_manager.remove(value)

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
                board.insert_cluster_manager(board, degree + 1)
                if degree > 1:  # Takes care of a cluster with a single value (does not allow deletion of it)
                    board.clusters.pop(val_left)
                board.remove_value_cluster_manager(board, degree)

        if i + 1 < len(board.deque):
            if board.deque[i + 1] == value + 1:
                val_right = board.deque[i + 1]
                opposite_lance, degree = board.clusters[val_right]
                if val_left != -1:  # We need to check if we have joined two clusters
                    if can_merge_cluster():
                        new_degree = degree + board.clusters[value][1]
                        board.clusters[opposite_lance] = (board.clusters[value][0], new_degree)
                        board.clusters[board.clusters[value][0]] = (opposite_lance, new_degree)
                        board.insert_cluster_manager(board, new_degree)
                        board.remove_value_cluster_manager(board, degree)
                        board.remove_value_cluster_manager(board, board.clusters[value][1])
                        board.clusters.pop(value)
                        if degree != 1:
                            board.clusters.pop(value + 1)
                        board.did_clusters_merge = True
                else:
                    board.clusters[value] = (opposite_lance, degree + 1)
                    board.clusters[opposite_lance] = (value, degree + 1)
                    board.insert_cluster_manager(board, degree + 1)
                    if degree > 1:  # Takes care of a cluster with a single value (does not allow deletion of it)
                        board.clusters.pop(val_right)
                    board.remove_value_cluster_manager(board, degree)

        # This is the first value being put in the deque structure
        if val_left == -1 and val_right == -1:
            board.clusters[value] = (value, 1)
            board.insert_cluster_manager(board, 1)  # Inits a cluster with a size of 1

    @staticmethod
    def get_deque_index(board, value):
        return bisect.bisect_left(board.deque, value)

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

        # TODO (ideias):
        #   -> Implement frontier nodes (done):
        #       -> Sempre que é  alterada uma coordenada da fronteira, as únicas fronteiras que mudam são as dos
        #          valores [val + 1, val - 1]
        #   -> Implement Priority Queue (no need):
        #       -> Vai usar uma tupla (len, (coordinates)) para obter a posição com menos valores possiveis
        #       -> Vai ter referencia para os valores com menos posições possiveis
        #   -> Implement Aglomerados e os pontos atribuidos
        #   -> Implement nós lança (nós nas pontas dos aglomerados das listas)

        # If it is the first iteration, we don't need to do anything because it has already been done
        if state.board.previous_action is None:
            return [(row, col, val) for (row, col), values in state.board.frontier.items() for val in values]

        # Unpacks action that was taken previously
        row, col, result = state.board.previous_action

        # Since we only need to update the values above and below (frontier property, we check if they are in the
        # board and update their adjacent frontier values
        if result + 1 in state.board.inserted:
            adj_row, adj_col = state.board.inserted[result + 1]
            Board.update_adjacent_frontier(state.board, adj_row, adj_col, result)
        if result - 1 in state.board.inserted:
            adj_row, adj_col = state.board.inserted[result - 1]
            Board.update_adjacent_frontier(state.board, adj_row, adj_col, result)

        # We also need to update our own adjacent frontier values
        up, down = state.board.adjacent_vertical_numbers(row, col)
        left, right = state.board.adjacent_horizontal_numbers(row, col)
        if up == 0:
            Board.clear_possibilities(state.board, row - 1, col)
            state.board.frontier[(row - 1, col)] = Board.get_possible_values(state.board, row - 1, col, state.board.inserted)
        if down == 0:
            Board.clear_possibilities(state.board, row + 1, col)
            state.board.frontier[(row + 1, col)] = Board.get_possible_values(state.board, row + 1, col, state.board.inserted)
        if left == 0:
            Board.clear_possibilities(state.board, row, col - 1)
            state.board.frontier[(row, col - 1)] = Board.get_possible_values(state.board, row, col - 1, state.board.inserted)
        if right == 0:
            Board.clear_possibilities(state.board, row, col + 1)
            state.board.frontier[(row, col + 1)] = Board.get_possible_values(state.board, row, col + 1, state.board.inserted)

        # TODO: improve this by storing the previous array and only add/remove the changes that were made to it
        return [(row, col, val) for (row, col), values in state.board.frontier.items() for val in values]

    def result(self, state, action):
        """Returns the resulting state of applying the input action (result from self.action(state)) to
        the current state."""
        new_state = copy.deepcopy(state)
        new_state.board.set_number(*action)
        new_state.board.inserted[action[2]] = (action[0], action[1])
        new_state.board.insert_deque(new_state.board, action[2])
        new_state.board.delete_possibility(new_state.board, action[2], new_state.board.frontier[(action[0], action[1])])
        new_state.board.frontier.pop((action[0], action[1]))
        new_state.board.previous_action = action
        return NumbrixState(new_state.board)

    def goal_test(self, state):
        """Checks if we have a valid solution of this game-"""

        if len(state.board.inserted) != state.board.max_value:
            return False

        # Stores the start of this board (number one)
        row, col = state.board.inserted[1]

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

    def h(self, node):
        """Heuristic function used in A*"""
        # TODO: ideas
        #       -> Ver se o valor que está ao meu lado é +/- 1 que eu e dar mais pontos
        #       -> Dar mais pontos à medida que vai formando um caminho (Muito lento)
        #       -> Somar linhas/colunas ??? (ideia do stor)
        #       -> Modelo epidemiológico?
        #       -> Calculo de distancias entre valores (valores mais proximos tem de estar mais perto e mais afastados
        #          tem de estar mais longe (a não ser que haja uma path forte)

        # TODO: Tips on the next implementation:
        #   -> Quanto mais preenchido estiver, mais valor lhe vou dar
        #   -> Só computar as actions uma vez para cada valor já no tabuleiro e depois apenas quando um dos seu vizinhos
        #      é alterado (vou ter de ter dois dict em que coloco os nós ainda não vistos aquando do result())
        #   -> Fechar os nós que já não têm mais opções (aquando do result(), posso ter uma estrutura para isso?)
        #   -> Ter uma matriz com todos os valores possiveis para cada possição. se algum array ficar vazio, dou um
        #      um péssimo valor na heuristica porque quer dizer que o tabuleiro n tem solução

        def calc_distance(value, target):
            upper = abs(value - target)
            value_row, value_col = node.state.board.inserted[value]
            target_row, target_col = node.state.board.inserted[target]
            lower = abs(value_row - target_row) + abs(value_col - target_col)
            result = upper / lower
            if result < 1:
                return 10000
            else:
                return result

        # Start of program does not need a heuristic value
        if node.action is None:
            return 1

        # Holds heuristic termination value
        KILLER_VALUE = 900000000000
        FRONTIER_LEN = 100
        IMPROVEMENT_FACTOR = 10000000
        SINGLE_CLUSTER_FACTOR = 10000
        SOLO_FACTOR = 500

        # Unpacks action to evaluate it
        row, col, val = node.action

        # Initializes the total value with the number of remaining positions
        total = (node.state.board.max_value - len(node.state.board.inserted))
        if len(node.state.board.clusters) == 2:
            total *= SINGLE_CLUSTER_FACTOR
        else:
            total *= IMPROVEMENT_FACTOR

        # Checks if this value only has one possible coordinate or if this action merged two clusters
        if node.parent.state.board.possibilities[val] == 1 or node.state.board.did_clusters_merge:
            node.state.board.did_clusters_merge = False
            return total - SOLO_FACTOR

        # Now, we evaluate the state of the cluster by giving more points to smaller clusters. This allows us to better
        # fill the "cracks" between each already placed value and once there is only a single cluster (a single path),
        # we just need to fill the other two ends
        if len(node.state.board.clusters) > 2:

            # Checks if this action is being taken in the smallest cluster. If not, we 'butcher' this action
            if node.state.board.clusters[val][1] - 1 <= Board.get_smallest_cluster_size(node.state.board):

                # Blocks expanding the deque to the ends while there are spaces between the clusters
                if node.parent.state.board.deque[0] - 1 == node.state.board.deque[0] \
                        or node.parent.state.board.deque[-1] + 1 == node.state.board.deque[-1]:
                    return KILLER_VALUE

                # Check if this coordinate has only one option and if so, we give a higher priority
                if len(node.parent.state.board.frontier[(row, col)]) == 1:
                    return total - FRONTIER_LEN // len(node.parent.state.board.frontier[(row, col)])

                # Gets leftmost and rightmost index from our deque structure (related to the inserted action) to
                # calculate the distance. This allows us to expand the board with plays that go according to a
                # radius of possible values and by using the 'lance' nodes, we can do this expansion towards the
                # next (lower or upper) value. We only consider actions that give distance = 1 because one of the
                # properties of the used formula is that it only returns 1 for EXACT and CORRECT positions in a
                # given action (only when filling 'cracks' between 2 clusters since this property is not valid
                # for end values in the deque because they don't have a "next node" to compare with)
                val_index = node.state.board.get_deque_index(node.state.board, val)
                if val_index + 1 < len(node.state.board.deque):  # Calculates right 'lance'
                    if calc_distance(val, node.state.board.deque[val_index + 1]) != 1:
                        return KILLER_VALUE
                if val_index - 1 >= 0:  # Calculates left 'lance'
                    if calc_distance(val, node.state.board.deque[val_index - 1]) != 1:
                        return KILLER_VALUE

                # If everything is fine, we return the calculated value
                return total - FRONTIER_LEN // len(node.parent.state.board.frontier[(row, col)])

            else:
                return KILLER_VALUE

        else:  # In this case, we are just trying to fill the ends in the deque structure

            # First thing to do is to check which side of the deque is smaller. We are going to start by filling it
            # because, when it is done, the other side is going to be a straight path to the solution (since there are
            # no more possible values)
            opposite, _ = node.state.board.clusters[val]
            if opposite > val:  # We are on the left side of the deque structure

                # Counts how many positions are free in either side
                right_side_remaining = node.state.board.max_value - opposite
                left_side_remaining = val - 1

                if left_side_remaining <= right_side_remaining or right_side_remaining == 0:  # Checks if we are using the lowest side

                    # Gives a better value if this position has less possible values
                    total += len(node.parent.state.board.frontier[(row, col)]) - (right_side_remaining + left_side_remaining)

                    # We give a better value for higher distances here because we want the value to be put further
                    # away from the other extreme
                    if right_side_remaining != 0:
                        total -= calc_distance(val, node.state.board.deque[-1])

                    return total

                else:
                    return KILLER_VALUE

            else:  # We are on the right side of the deque structure

                # Counts how many positions are free in either side
                right_side_remaining = node.state.board.max_value - val
                left_side_remaining = opposite - 1

                if right_side_remaining <= left_side_remaining or left_side_remaining == 0:  # Checks if we are using the lowest side

                    # Gives a better value if this position has less possible values
                    total += len(node.parent.state.board.frontier[(row, col)]) - (right_side_remaining + left_side_remaining)

                    # We give a better value for higher distances here because we want the value to be put further
                    # away from the other extreme
                    if left_side_remaining != 0:
                        total -= calc_distance(val, node.state.board.deque[0])

                    return total

                else:
                    return KILLER_VALUE


if __name__ == "__main__":

    # Creates matrix that represents game board from input file
    instance = Board(Board.parse_instance(sys.argv[1]))

    # Initializes Numbrix problem
    numbrix = Numbrix(instance)

    # Applies our search algorithm to find the correct solution
    solved = astar_search(numbrix).state.board.get_result()

    # Shows result in stdin
    print(solved, end="")
