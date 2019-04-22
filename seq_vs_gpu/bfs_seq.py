# sequential bfs: finding path to a goal node
from collections import defaultdict
import time
import sys

# graph class
class Graph: 
  
    def __init__(self):  
        self.graph = defaultdict(list)  # list storage
        self.prev = defaultdict(list)   # previous vertex in path
  
    def addEdge(self, u, v): 
        self.graph[u].append(v)
        self.graph[v].append(u)

    def printPath (self, s, g):
        path = ""
        while s != g:
            path = " -> " + str(g) + path
            g = self.prev[g]
        path = str(s) + path
        print path
  
    # bfs algorithm
    def BFS(self, s, g): 
        visited = defaultdict(list)
        for k, v in self.graph.iteritems():   # no vertices visited
            visited[k] = False
        queue = [] 
        queue.append(s)                       # source visited
        visited[s] = True
  
        while queue:
            c = queue.pop(0)
            if c == g:
                self.printPath(s, g)
                return
            
            for i in self.graph[c]: 
                if visited[i] == False:
                    queue.append(i) 
                    visited[i] = True
                    self.prev[i] = c          # previous vertex is parent
        print "Not Found", g

if __name__ == "__main__":

    # validation
    if (len(sys.argv) != 4):
        print 'Error: Usage: python bfs_seq.py <edges_file>:str <source_vertex>:int <goal_vertex>:int'
        sys.exit()

    g = Graph()
    # file with list of edges
    with open("dataSets/" + sys.argv[1], "r") as edges:
        for e in edges:
            vertices = e.split()
            g.addEdge(int(vertices[0]), int(vertices[1]))

    start = time.time()
    g.BFS(int(sys.argv[2]), int(sys.argv[3]))
print time.time() - start