# gpu bfs: traversing through all the nodes
import pyopencl as cl
import pyopencl.tools
import pyopencl.array
import numpy as np
import os
import sys
from collections import defaultdict
import time

if __name__ == "__main__":
    os.environ['PYOPENCL_CTX'] = '1'
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    # platform, device, context & queue setup
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    # validation
    if (len(sys.argv) != 3):
        print 'Error: Usage: python bfs_gpu_spread.py <no_of_threads>:int <edges_file>:str'
        sys.exit()

    TASKS = int(sys.argv[1])

    # grab all edges from input file
    graph = defaultdict(list)
    nbr_list = []
    with open("dataSets/" + sys.argv[2], "r") as edges:
        for e in edges:
            vertices = e.split()
            graph[int(vertices[0])].append(int(vertices[1]))

    # constructing list of neighbours
    vertex = np.dtype([("id", np.int32), ("size", np.int32), ("start", np.int32)])
    struct_arr = np.empty(len(graph), vertex)

    fill = 0
    num_nbrs = 0
    for v, nbrs in graph.items():
        struct_arr[fill]["id"] = v
        struct_arr[fill]["size"] = len(nbrs)
        struct_arr[fill]["start"] = num_nbrs
        for nbr in nbrs:
            nbr_list.append(nbr)
            num_nbrs += 1
        fill += 1
    
    nbr_list = np.array(nbr_list)
    nbr_list = nbr_list.astype(np.int32)

    # bfs stuff
    visited = np.full(len(graph), -1)  # visited vertices storage
    vertices = np.array([len(graph)])  # number of vertices
    traversed = np.full(1, 0)          # number of vertices in traversed chunk

    # dividing task among threads
    sources = []
    step = 0
    for k, v in graph.items():
        sources.append(k)
        step += 1
        if step == TASKS:
            break
    sources = np.array(sources)
    sources = sources.astype(np.int32)
    
    # transfer struct array to device
    dev_struct_arr = cl.array.to_device(queue, struct_arr)

    # buffer allocation
    mf = cl.mem_flags
    nbr_list_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nbr_list)
    visited_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=visited)
    vertices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertices)
    sources_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sources)
    traversed_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=traversed)

    # kernel program
    prg = cl.Program(ctx, '''
        typedef struct {
            int id;
            int size;
            int start;
        } vertex;

        __kernel void bfs (__global vertex structs[], __global int nbr_list[], __global int visited[], __global int vertices[], __global int sources[], __global int traversed[]) {
            int gid = get_global_id(0);
            int v = 0;
            int fsize = 1;
            int frontier[800];
            for (int j=0; j<vertices[0]; j++)
                frontier[j] = -1;
            frontier[0] = sources[gid];
            int current;
            
            while (fsize>0) {               
                current = frontier[0];
                for (int fr=0; fr<fsize; fr++)
                    frontier[fr] = frontier[fr+1];
                frontier[fsize-1] = -1;
                fsize--;
                visited[v++] = current;
                printf("%d ", current);
                              
                for (int s=0; s<vertices[0]; s++) {
                    if (structs[s].id == current) {
                        for (int n=structs[s].start; n<structs[s].start + structs[s].size; n++)  {
                            int nb = nbr_list[n];
                            int explored = 0;
                            for(int vs=0; vs<v; vs++) {
                                if (nb == visited[vs]) {
                                    explored = 1;
                                    break;
                                }
                            }
                            for (int f=0; f<fsize; f++) {
                                if (nb == frontier[f]) {
                                    explored = 1;
                                    break;
                                }
                            }

                            if (explored == 1)
                                continue;
                            else
                                frontier[fsize++] = nb;
                        }
                        break;
                    }
                }
            }
            traversed[0] = v;
        }
    ''').build()
    

    start = time.time()
    event = prg.bfs(queue, (TASKS,), (1,), dev_struct_arr.data, nbr_list_buf, visited_buf, vertices_buf, sources_buf, traversed_buf)
    event.wait()

    v = np.empty_like(traversed)
    cl.enqueue_copy(queue, v, traversed_buf)

    print '\nVertices traversed: ', v[0]
    print 'Existent vertices: ', len(graph)
    print 'Time taken: ', time.time() - start