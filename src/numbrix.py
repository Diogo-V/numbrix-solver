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
        self.inserted, self.frontier = self.build_matrix_structs()

    def build_matrix_structs(self):
        """Builds a dictionary with the values already in the board + their coordinates and also setups the frontier."""

        def try_check(row, col, matrix, matrix_col_size):
            if 0 <= row < matrix_col_size and 0 <= col < matrix_col_size:
                if matrix[row][col] == 0:
                    if (row, col) not in frontier:
                        possible_values = Board.is_frontier(self, row, col, inserted)
                        frontier[(row, col)] = possible_values

        inserted = {}
        frontier = {}

        # Goes over all positions in the matrix and checks if they are already filled and stores their coordinates
        for i in range(self.n):
            for j in range(self.n):
                val = self.get_number(i, j)
                if val != 0:

                    inserted[val] = (i, j)

                    # Stores possible values for the adjacent nodes
                    try_check(i + 1, j, self.matrix, self.n)
                    try_check(i - 1, j, self.matrix, self.n)
                    try_check(i, j + 1, self.matrix, self.n)
                    try_check(i, j - 1, self.matrix, self.n)

        return inserted, frontier

    @staticmethod
    def is_frontier(board, row, col, inserted):
        """Checks if coordinate is frontier and if so, stores its possible values and returns them."""

        def check(border):
            result1 = []
            if border is not None and border != 0:
                if border - 1 not in inserted and 0 < border - 1 < board.max_value:
                    result1.append(border - 1)
                if border + 1 not in inserted and 0 < border + 1 < board.max_value:
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

        # TODO:
        #   -> Implement frontier nodes
        #   -> Implement Priority Queue
        #   -> Implement Aglomerados e os pontos atribuidos
        #   -> Implement nós lança (nós nas pontas dos aglomerados das listas)

        # Holds list of actions that can be taken
        result = []

        # Holds the frontier for the next iteration
        frontier = {}

        # Iterates over all the inserted items and returns the possible action for each one of them
        for key, (row, col) in state.board.frontier.items():  # TODO: fix this

            # Gets restriction values adjacent to the currently being evaluated position
            up, down = state.board.adjacent_vertical_numbers(row, col)
            left, right = state.board.adjacent_horizontal_numbers(row, col)

            # Gets available adjacent values
            available = [i for i in [key + 1, key - 1] if i not in state.board.inserted and 0 < i < state.board.max_value]

            # Iterates over the possible adjacent values and appends possible actions to the result list
            for val in available:

                # We need to check if our adjacency is an empty position before appending the action. Also adds this
                # values to the not yet visited state
                if up == 0:
                    action = (row - 1, col, val)
                    result.append(action)
                    frontier[val] = (row - 1, col)
                if down == 0:
                    action = (row + 1, col, val)
                    result.append(action)
                    frontier[val] = (row + 1, col)
                if left == 0:
                    action = (row, col - 1, val)
                    result.append(action)
                    frontier[val] = (row, col - 1)
                if right == 0:
                    action = (row, col + 1, val)
                    result.append(action)
                    frontier[val] = (row, col + 1)

        # Sets the not visited structure with the newly appended values since they have not yet been seen
        state.board.not_visited = frontier

        return result

    def result(self, state, action):
        """Returns the resulting state of applying the input action (result from self.action(state)) to
        the current state."""
        new_state = copy.deepcopy(state)
        new_state.board.set_number(*action)
        state.board.inserted[action[2]] = (action[0], action[1])
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

        # TODO: Tips on the next implementation:
        #   -> os dict's em python estam ordenados, por isso, basta-me ver se estou a ir na posição do i-1 e i+1
        #   -> Quanto mais preenchido estiver, mais valor lhe vou dar
        #   -> Ao calcular as actions(), podem criar uma priority queue que tem em primeiro lugar os valores que têm
        #      menos opções para serem colocados no tabuleiro. depois, usamos isto logo aqui na função
        #   -> Só computar as actions uma vez para cada valor já no tabuleiro e depois apenas quando um dos seu vizinhos
        #      é alterado (vou ter de ter dois dict em que coloco os nós ainda não vistos aquando do result())
        #   -> Fechar os nós que já não têm mais opções (aquando do result(), posso ter uma estrutura para isso?)
        #   -> Contar o nr de casas vazias por linha e somar ao valor que a heuristica dá

        def check_opposite_side(val1, up1, down1, left1, right1, already_checked):
            if up1 is not None and up1 != 0 and up1 not in visited and up1 != already_checked:
                if val1 == up1 + 1 or val1 == up1 - 1:
                    return up1, row - 1, col

            if down1 is not None and down1 != 0 and down1 not in visited and down1 != already_checked:
                if val1 == down1 + 1 or val1 == down1 - 1:
                    return down1, row + 1, col

            if left1 is not None and left1 != 0 and left1 not in visited and left1 != already_checked:
                if val1 == left1 + 1 or val1 == left1 - 1:
                    return left1, row, col - 1

            if right1 is not None and right1 != 0 and right1 not in visited and right1 != already_checked:
                if val1 == right1 + 1 or val1 == right1 - 1:
                    return right1, row, col + 1

            return ()

        # All nodes are initialized with a very big number. This is done because we want to prioritize lower heuristic
        # values over bigger ones
        total = 1000000000

        # We initialize our loop condition a true to be able to find a path
        found_path = True

        # This value is going to be used to "reward" a good path. Each time we find a path, this value is used to
        # subtract from the total amount. This allows us to be able to "reward" longer paths
        reward = 2

        # Holds list of visited values (used to not repeat already visited positions)
        visited = []

        # If this node does not have an action applied to it, then we don't need to find a path
        if node.action is None:
            return total

        # Gets action that is going to be taken on this node and unpacks it to get the adjacent nodes
        row, col, val = node.action

        # Holds iteration value
        current_val = val

        # Checks if we are seeing a value in the middle of a path and if so, we need to store the other side's info and
        # proceed to the opposite one
        has_opposite = False
        iteration = 0
        opp_val = -1
        opp_row = -1
        opp_col = -1

        while found_path:

            # Gets adjacent values
            up, down = node.state.board.adjacent_vertical_numbers(row, col)
            left, right = node.state.board.adjacent_horizontal_numbers(row, col)

            # Now, we try to form a path from this node by evaluating all adjacent nodes and see if they form a path too
            # and if so, we update the current value and double the next score if we can find even more nodes that form
            # a path
            if up is not None and up != 0 and up not in visited:
                if current_val == up + 1 or current_val == up - 1:
                    opposite = check_opposite_side(current_val, up, down, left, right, up) if iteration == 0 else ()
                    iteration += 1
                    if opposite != ():
                        has_opposite = True
                        opp_val, opp_row, opp_col = opposite
                    total -= reward
                    reward *= 2
                    row -= 1
                    visited.append(current_val)
                    current_val = up
                    continue

            if down is not None and down != 0 and down not in visited:
                if current_val == down + 1 or current_val == down - 1:
                    opposite = check_opposite_side(current_val, up, down, left, right, down) if iteration == 0 else ()
                    iteration += 1
                    if opposite != ():
                        has_opposite = True
                        opp_val, opp_row, opp_col = opposite
                    total -= reward
                    reward *= 2
                    row += 1
                    visited.append(current_val)
                    current_val = down
                    continue

            if left is not None and left != 0 and left not in visited:
                if current_val == left + 1 or current_val == left - 1:
                    opposite = check_opposite_side(current_val, up, down, left, right, left) if iteration == 0 else ()
                    iteration += 1
                    if opposite != ():
                        has_opposite = True
                        opp_val, opp_row, opp_col = opposite
                    total -= reward
                    reward *= 2
                    col -= 1
                    visited.append(current_val)
                    current_val = left
                    continue

            if right is not None and right != 0 and right not in visited:
                if current_val == right + 1 or current_val == right - 1:
                    opposite = check_opposite_side(current_val, up, down, left, right, right) if iteration == 0 else ()
                    iteration += 1
                    if opposite != ():
                        has_opposite = True
                        opp_val, opp_row, opp_col = opposite
                    total -= reward
                    reward *= 2
                    col += 1
                    visited.append(current_val)
                    current_val = right
                    continue

            if has_opposite:  # Checks if we had more to the other side
                current_val = opp_val
                row = opp_row
                col = opp_col
                total -= reward
                reward *= 2
                has_opposite = False
                continue

            # In case no valid path is found, then we stop this loop and return the total accumulated heuristic value
            found_path = False

        return total


if __name__ == "__main__":

    # Creates matrix that represents game board from input file
    board = Board(Board.parse_instance(sys.argv[1]))

    # Initializes Numbrix problem
    numbrix = Numbrix(board)

    # Applies our search algorithm to find the correct solution
    # result = astar_search(numbrix).state.board.get_result()
    result = depth_first_tree_search(numbrix).state.board.get_result()

    # Shows result in stdin
    print(result, end="")
