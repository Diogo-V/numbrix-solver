# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 06:
# 95555 Diogo Venâncio
# 95675 Sofia Morgado

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
    """ Representação interna de um tabuleiro de Numbrix. """

    def __init__(self, init_matrix: list[list[int]]):
        self.n = len(init_matrix)
        self.matrix = init_matrix
    
    def get_number(self, row: int, col: int) -> int:
        """ Devolve o valor na respetiva posição do tabuleiro. """
        return self.matrix[row][col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """ Devolve os valores imediatamente abaixo e acima, 
        respectivamente. """
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

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """ Devolve os valores imediatamente à esquerda e à direita, 
        respectivamente. """
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
    
    @staticmethod    
    def parse_instance(filename: str) -> list[list[int]]:
        """ Lê o ficheiro cujo caminho é passado como argumento e retorna
        uma instância da classe Board. """
        with open(filename, encoding="utf-8") as f:

            # Gets column and row sizes
            n = int(f.readline())

            # Puts rest of the lines in a matrix to represents the board
            return [[int(word) for word in f.readline().split() if word.isdigit()] for _ in range(n)]


class Numbrix(Problem):
    def __init__(self, board: Board):
        """ O construtor especifica o estado inicial. """
        # TODO
        pass

    def actions(self, state: NumbrixState):
        """ Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. """
        # TODO
        pass

    def result(self, state: NumbrixState, action):
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de 
        self.actions(state). """
        # TODO
        pass

    def goal_test(self, state: NumbrixState):
        """ Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro 
        estão preenchidas com uma sequência de números adjacentes. """
        # TODO
        pass

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        # TODO
        pass
    

if __name__ == "__main__":

    # Ler o ficheiro de input de sys.argv[1],
    if len(sys.argv) == 2:

        # Creates matrix that represents game board
        board = Board(Board.parse_instance(sys.argv[1]))

        # Usar uma técnica de procura para resolver a instância,
        # Retirar a solução a partir do nó resultante,
        # Imprimir para o standard output no formato indicado.
        pass

    else:
        print("Invalid number of arguments. Only needs a path to be passed!")
