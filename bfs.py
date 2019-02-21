# breadth-frst search algorithm for graph of +ve integers
# after one kernel item is done finding the goal or done searching the whole graph, halts every other work item
import pyopencl as cl
import numpy as np
import os
import sys

TASKS = 3

if __name__ == "__main__":
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    os.environ['PYOPENCL_CTX'] = '1'

    # platform, device, context & queue setup
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    # graph attributes
    keys = np.array([0, 2, 3, 4, 5, 6, 7, 8, 1]) # vertex ids
    keys = keys.astype(np.int32)

    graph = np.array([ # maintaining a fixed row size filled with -1 as non-existent neighbours
        2, 7, -1, -1,
        0, 3, -1, -1,
        2, 6, -1, -1,
        1, 5, 6, 7, 
        4, 6, 7, -1,
        3, 4, -1, -1,
        0, 5, 8, -1,
        1, 7, -1, -1,
        4, 8, -1, -1
    ])
    graph = graph.astype(np.int32)

    neighbours = np.array([2, 2, 2, 3, 3, 2, 2, 2, 2]) # of neighbours of each vertex
    neighbours = neighbours.astype(np.int32)

    visited = np.empty(keys.size)      # visited vertices
    frontier = np.full(keys.size, -1)  # next in line to be traversed with arbitrary source
    goal = np.array([6])               # vertex to find
    vertices = np.array([keys.size])   # number of vertices
    max_neighbours = np.array([4])     # maximum neighbours possible for a vertex
    found = np.array([0])              # search result indicator

    # dividing task among threads
    threads = []
    first_neighbours = []

    step = keys.size/TASKS             # vertices per thread
    for g in range(0, TASKS):
        threads.append(keys[g*step])   # assigning scattered starting points
    threads = np.array(threads)
    threads = threads.astype(np.int32)
    
    # buffer allocation
    mf = cl.mem_flags
    graph_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=graph)
    keys_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys)
    neighbours_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=neighbours)
    visited_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=visited)
    frontier_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=frontier)
    goal_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=goal)
    vertices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertices)
    max_neighbours_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=max_neighbours)
    threads_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=threads)
    found_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found)

    # kernel program
    prg = cl.Program(ctx, '''
        __kernel void bfs (__global int graph[], __global int keys[], __global int neighbours[], __global int visited[], __global int frontier[], __global int* goal, __global int* vertices, __global int* max_neighbours, __global int* threads, __global int* found) {
            int gid = get_global_id(0);
            frontier[0] = *(threads+gid);
            int fsize = 1;
            int vsize = 0;
            int max_n = *max_neighbours;
            while (true) {
                if (found[0] > 0)
                    return;
                // printf("Frontier:");
                for (int i=0; i<fsize; i++) {
                    // printf(" %d", frontier[i]);
                }
                // printf("\\n");
                if (fsize == 0) {
                    found[0]++;
                    printf("Not found %d\\n", goal[0]);
                    return;
                }
                int current = *(frontier + 0);
                if (current == *goal) {
                    found[0]++;
                    printf("Found %d\\n", current);
                    return;
                }
                for (int fr=0; fr<fsize-1; fr++) {
                    frontier[fr] = frontier[fr+1];
                }
                *(frontier + fsize - 1) = -1;
                fsize--;
                *(visited + vsize) = current;
                vsize++;
                // printf("Visited:");
                // for (int i=0; i<vsize; i++) {
                //    printf(" %d", visited[i]);
                // }
                // printf("\\n");
                
                for (int s=0; s<*vertices; s++) {
                    if (keys[s] == current) {
                        for (int n=0; n<neighbours[s]; n++)  {
                            int nb = graph[s*max_n + n];
                            int already = 0;
                            for(int v=0; v<vsize; v++) {
                                if (nb == *(visited+v)) {
                                    already = 1;
                                    break;
                                }
                            }
                            if (already == 1) {
                                continue;
                            }
                            for (int f=0; f<fsize; f++) {
                                if (nb == *(frontier+f)) {
                                    already = 1;
                                    break;
                                }
                            }
                            if (already == 0) {
                                *(frontier + fsize) = nb;
                                fsize++;
                            }
                        }
                        // printf("\\n");
                        break;
                    }
                    if (found[0] > 0)
                        return;
                }
            }
        }
    ''').build()

    prg.bfs(queue, (TASKS,), (1,), graph_buf, keys_buf, neighbours_buf, visited_buf, frontier_buf, goal_buf, vertices_buf, max_neighbours_buf, threads_buf, found_buf)