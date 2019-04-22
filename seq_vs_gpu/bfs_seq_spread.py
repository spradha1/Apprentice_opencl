# sequential bfs: traversing through all the nodes
from collections import defaultdict
import time
import sys

# graph class
class Graph: 
  
    def __init__(self):  
        self.graph = defaultdict(list)  # list storage
        self.visited = defaultdict(list)
  
    def addEdge(self, u, v): 
        self.graph[u].append(v)
        self.graph[v].append(u)
  
    # bfs algorithm
    def BFS(self, src): 
        for k, v in self.graph.iteritems():   # no vertices visited
            self.visited[k] = False
        queue = [] 
        queue.append(src)                     # source visited
        v = 0
  
        while queue:
            c = queue.pop(0)
            self.visited[c] = True
            print str(c) + " ",
            v += 1
            for i in self.graph[c]:
                if not self.visited[i] and i not in queue:
                    queue.append(i)
        print '\nVertices traversed: ', v

if __name__ == "__main__": 
    # validation
    if (len(sys.argv) != 3):
        print 'Error: Usage: python bfs_seq_spread.py <edges_file>:str <source_vertex>:int'
        sys.exit()

    g = Graph()
    # file with list of edges
    with open("dataSets/" + sys.argv[1], "r") as edges:
        for e in edges:
            vertices = e.split()
            g.addEdge(int(vertices[0]), int(vertices[1]))
    
    start = time.time()
    g.BFS(int(sys.argv[2]))                   # source vertex
    print 'Existent vertices: ', len(g.graph)
    print 'Time taken: ', time.time() - start