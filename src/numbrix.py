# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 06:
# 95555 Diogo Venâncio
# 95675 Sofia Morgado
import copy
import sys
import heapq as PQ

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
        self.queue = []
        PQ.heapify(self.queue)
        self.inserted, self.frontier = self.build_matrix_structs()
        self.previous_action = None

    def build_matrix_structs(self):
        """Builds a dictionary with the values already in the board + their coordinates and also setups the frontier."""

        def build_frontier(row, col, board):
            if 0 <= row < board.n and 0 <= col < board.n:
                if board.matrix[row][col] == 0:
                    if (row, col) not in frontier:
                        possible_values = Board.get_possible_values(self, row, col, inserted)
                        frontier[(row, col)] = possible_values
                        PQ.heappush(board.queue, (len(possible_values), (row, col)))

        # Goes over all positions in the matrix and checks if they are already filled and stores their coordinates
        inserted = {}
        for i in range(self.n):
            for j in range(self.n):
                val = self.get_number(i, j)
                if val != 0:
                    inserted[val] = (i, j)

        # Builds frontiers after knowing which nodes have been inserted
        frontier = {}
        for (i, j) in inserted.values():
            build_frontier(i + 1, j, self)
            build_frontier(i - 1, j, self)
            build_frontier(i, j + 1, self)
            build_frontier(i, j - 1, self)

        return inserted, frontier

    @staticmethod
    def get_possible_values(board, row, col, inserted):
        """Returns possible values for input coordinate. Returns empty list if none are found."""

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

    @staticmethod
    def update_adjacent_frontier(board, row, col, value):
        """Updates board frontier values when 'value' is inserted in input coordinates."""

        # Gets currently open adjacent coordinates (these are the only ones that can get updated)
        up, down = board.adjacent_vertical_numbers(row, col)
        left, right = board.adjacent_horizontal_numbers(row, col)

        if up == 0:
            board.remove_frontier(board, row - 1, col, value)
            # TODO: update queue
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
        except ValueError:
            pass

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
        #   -> Implement Priority Queue (done):
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
        if up == 0:  # TODO: update queue
            state.board.frontier[(row - 1, col)] = Board.get_possible_values(state.board, row - 1, col, state.board.inserted)
        if down == 0:
            state.board.frontier[(row + 1, col)] = Board.get_possible_values(state.board, row + 1, col, state.board.inserted)
        if left == 0:
            state.board.frontier[(row, col - 1)] = Board.get_possible_values(state.board, row, col - 1, state.board.inserted)
        if right == 0:
            state.board.frontier[(row, col + 1)] = Board.get_possible_values(state.board, row, col + 1, state.board.inserted)

        # TODO: improve this by storing the previous array and only add/remove the changes that were made to it
        return [(row, col, val) for (row, col), values in state.board.frontier.items() for val in values]

    def result(self, state, action):
        """Returns the resulting state of applying the input action (result from self.action(state)) to
        the current state."""
        new_state = copy.deepcopy(state)
        new_state.board.set_number(*action)
        new_state.board.inserted[action[2]] = (action[0], action[1])
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
        #   -> Ter uma matriz com todos os valores possiveis para cada possição. se algum array ficar vazio, dou um
        #      um péssimo valor na heuristica porque quer dizer que o tabuleiro n tem solução

        return 1


if __name__ == "__main__":

    # Creates matrix that represents game board from input file
    instance = Board(Board.parse_instance(sys.argv[1]))

    # Initializes Numbrix problem
    numbrix = Numbrix(instance)

    # Applies our search algorithm to find the correct solution
    # solved = astar_search(numbrix).state.board.get_result()
    solved = depth_first_tree_search(numbrix).state.board.get_result()

    # Shows result in stdin
    print(solved, end="")
