# É necessária a instalação das bibliotecas abaixo para o devido funcionamento do programa.
# É aconselhável maximizar as janelas com as plotagens dos resultados para melhor visualização.

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq
import time

# Definições do ambiente para BFS
class Environment:
    def __init__(self, shape, start, goal, num_obstacles=100):
        self.shape = shape  # formato do ambiente (x, y, z)
        self.start = start  # ponto inicial (x, y, z)
        self.goal = goal    # ponto final (x, y, z)
        self.grid = np.zeros(shape)  # cria uma grade 3D representando o ambiente
        self.visited = np.zeros(shape, dtype=bool)  # matriz para marcar células já visitadas
        self.add_obstacles(num_obstacles)  # Adiciona os obstáculos

    # Adiciona obstáculos aleatórios ao ambiente
    def add_obstacles(self, num_obstacles):
        for _ in range(num_obstacles):
            while True:
                # Gera posições aleatórias para os obstáculos
                obstacle_position = (np.random.randint(self.shape[0]),
                                     np.random.randint(self.shape[1]),
                                     np.random.randint(self.shape[2]))
                # Garante que a posição do obstáculo não seja o ponto inicial ou final
                if obstacle_position != self.start and obstacle_position != self.goal:
                    self.grid[obstacle_position] = 1  # Marca a célula como obstáculo
                    break

    # Verifica se um ponto é válido (dentro dos limites, não visitado e não é obstáculo)
    def is_valid(self, point):
        x, y, z = point
        return (0 <= x < self.shape[0] and
                0 <= y < self.shape[1] and
                0 <= z < self.shape[2] and
                not self.visited[x, y, z] and
                self.grid[x, y, z] == 0)

# Função para visualizar a execução de um caminho
def visualize_path(env, path, title, algorithm_label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{title}")  # Define o título com o rótulo do algoritmo

    # Cria uma grade 3D para visualização
    ax.scatter(*np.indices(env.shape).reshape(3, -1), c='lightblue', alpha=0.3)

    # Exibe os obstáculos na cor preta
    obstacle_indices = np.argwhere(env.grid == 1)
    ax.scatter(*obstacle_indices.T, c='black', s=100, label='Obstáculos')

    # Exibe o ponto inicial (verde) e final (vermelho)
    ax.scatter(*env.start, c='green', s=100, label='Início')
    ax.scatter(*env.goal, c='red', s=100, label='Fim')

    # Exibe o caminho encontrado, se houver
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], color='blue', linewidth=2, label='Caminho')

    ax.legend()
    # Exibe o rótulo do algoritmo na parte superior da visualização
    ax.text2D(0.05, 0.95, f"Algoritmo: {algorithm_label}", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.show()

# Classe para representar os nós
class Node:
    def __init__(self, x, y, z, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent  # Nó pai, usado para reconstruir o caminho
        self.g = 0  # Custo do caminho desde o início até o nó atual
        self.h = 0  # Heurística: estimativa de distância até o objetivo
        self.f = 0  # f = g + h (custo total estimado)

    def __lt__(self, other):
        return self.f < other.f  # Define a comparação entre nós com base no custo f

# Função para realizar a busca em largura (BFS)
def bfs_search(env):
    start_node = Node(*env.start)  # Cria o nó inicial
    queue = deque([start_node])  # Fila de nós a serem explorados
    env.visited[env.start] = True  # Marca o nó inicial como visitado
    path = []  # Lista para armazenar o caminho
    nodes_visited = 0  # Contador de nós percorridos

    while queue:
        current_node = queue.popleft()  # Remove o nó da frente da fila
        nodes_visited += 1  # Incrementa o número de nós percorridos

        # Se o nó atual é o objetivo, reconstrói o caminho
        if (current_node.x, current_node.y, current_node.z) == env.goal:
            while current_node is not None:
                path.append((current_node.x, current_node.y, current_node.z))
                current_node = current_node.parent
            print(f"Total de nós percorridos pelo BFS: {nodes_visited}")
            return path[::-1]  # Retorna o caminho em ordem reversa

        # Explora os vizinhos do nó atual
        for direction in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            neighbor_position = (current_node.x + direction[0],
                                 current_node.y + direction[1],
                                 current_node.z + direction[2])
            if env.is_valid(neighbor_position):  # Verifica se o vizinho é válido
                neighbor = Node(*neighbor_position, parent=current_node)
                queue.append(neighbor)  # Adiciona o vizinho à fila
                env.visited[neighbor_position] = True  # Marca o vizinho como visitado

    print(f"Total de nós percorridos pelo BFS: {nodes_visited}")
    
    return path

# Função para realizar a busca em profundidade (DFS)
def dfs_search(space_3d, start, goal):
    stack = [Node(*start)]  # Pilha de nós a serem explorados
    visited = set()  # Conjunto de nós já visitados
    nodes_visited = 0  # Contador de nós percorridos

    while stack:
        current_node = stack.pop()  # Remove o nó do topo da pilha
        visited.add((current_node.x, current_node.y, current_node.z))  # Marca como visitado
        nodes_visited += 1  # Incrementa o número de nós percorridos

        # Se o nó atual é o objetivo, reconstrói o caminho
        if (current_node.x, current_node.y, current_node.z) == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y, current_node.z))
                current_node = current_node.parent
            print(f"Total de nós percorridos pelo DFS: {nodes_visited}")
            return path[::-1]  # Retorna o caminho em ordem reversa

        # Explora os vizinhos do nó atual
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            neighbor_pos = (current_node.x + dx, current_node.y + dy, current_node.z + dz)
            # Verifica se o vizinho está dentro dos limites do ambiente e não é um obstáculo
            if not (0 <= neighbor_pos[0] < space_3d.shape[0] and
                    0 <= neighbor_pos[1] < space_3d.shape[1] and
                    0 <= neighbor_pos[2] < space_3d.shape[2]):
                continue

            if space_3d[neighbor_pos[0], neighbor_pos[1], neighbor_pos[2]] == 1:
                continue

            neighbor = Node(*neighbor_pos, parent=current_node)
            if (neighbor.x, neighbor.y, neighbor.z) in visited:  # Se já foi visitado, ignora
                continue

            stack.append(neighbor)  # Adiciona o vizinho à pilha

    print(f"Total de nós percorridos pelo DFS: {nodes_visited}")
    return []

# Função para realizar a busca A estrela (A*)
def a_star_search(space_3d, start, goal):
    open_list = []  # Lista de nós a serem explorados
    closed_list = set()  # Conjunto de nós já explorados
    nodes_visited = 0  # Contador de nós percorridos

    start_node = Node(*start)  # Cria o nó inicial
    goal_node = Node(*goal)  # Cria o nó objetivo
    heapq.heappush(open_list, start_node)  # Adiciona o nó inicial à lista de exploração

    while open_list:
        current_node = heapq.heappop(open_list)  # Remove o nó com o menor custo f
        closed_list.add((current_node.x, current_node.y, current_node.z))  # Marca como visitado
        nodes_visited += 1  # Incrementa o número de nós percorridos

        # Se o nó atual é o objetivo, reconstrói o caminho
        if (current_node.x, current_node.y, current_node.z) == (goal_node.x, goal_node.y, goal_node.z):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y, current_node.z))
                current_node = current_node.parent
            print(f"Total de nós percorridos pelo A*: {nodes_visited}")
            return path[::-1]  # Retorna o caminho em ordem reversa

        # Explora os vizinhos do nó atual
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            neighbor_pos = (current_node.x + dx, current_node.y + dy, current_node.z + dz)
            # Verifica se o vizinho está dentro dos limites do ambiente e não é um obstáculo
            if not (0 <= neighbor_pos[0] < space_3d.shape[0] and
                    0 <= neighbor_pos[1] < space_3d.shape[1] and
                    0 <= neighbor_pos[2] < space_3d.shape[2]):
                continue

            if space_3d[neighbor_pos[0], neighbor_pos[1], neighbor_pos[2]] == 1:
                continue

            neighbor = Node(*neighbor_pos, parent=current_node)
            if (neighbor.x, neighbor.y, neighbor.z) in closed_list:  # Se já foi visitado, ignora
                continue

            # Calcula custos g (real) e h (heurística) do vizinho
            neighbor.g = current_node.g + 1
            neighbor.h = abs(neighbor.x - goal_node.x) + abs(neighbor.y - goal_node.y) + abs(neighbor.z - goal_node.z)
            neighbor.f = neighbor.g + neighbor.h  # Custo total f

            heapq.heappush(open_list, neighbor)  # Adiciona o vizinho à lista de exploração

    print(f"Total de nós percorridos pelo A*: {nodes_visited}")
    return []

# Definindo o ambiente e executando as buscas
shape = (10, 10, 10)
start = (0, 0, 0)
goal = (9, 9, 9)
env = Environment(shape, start, goal)

# Executar BFS
start_time = time.time()
bfs_path = bfs_search(env)
end_time = time.time()
if bfs_path:
    print(f"BFS - Nós no caminho: {len(bfs_path)}")
visualize_path(env, bfs_path, "Caminho do BFS", "BFS")
print(f"Tempo de Execução do BFS: {end_time - start_time} segundos")

# Executar A*
start_time = time.time()
a_star_path = a_star_search(env.grid, start, goal)
end_time = time.time()
if a_star_path:
    print(f"A* - Nós no caminho: {len(a_star_path)}")
visualize_path(env, a_star_path, "Caminho do A*", "A*")
print(f"Tempo de Execução do A*: {end_time - start_time} segundos")

# Executar DFS
start_time = time.time()
dfs_path = dfs_search(env.grid, start, goal)
end_time = time.time()
if dfs_path:
    print(f"DFS - Nós no caminho: {len(dfs_path)}")
visualize_path(env, dfs_path, "Caminho do DFS", "DFS")
print(f"Tempo de Execução do DFS: {end_time - start_time} segundos")
