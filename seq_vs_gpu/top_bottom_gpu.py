# gpu bfs: finding path to a goal node using top-down and bottom-up approach
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
    if (len(sys.argv) != 4):
        print 'Error: Usage: python top_bottom_gpu.py <no_of_threads>:int <edges_file>:str <goal_vertex>:int'
        sys.exit()

    # grab all edges from input file
    graph = defaultdict(list)
    prev = defaultdict(list)
    nbr_list = []
    with open("dataSets/" + sys.argv[2], "r") as edges:
        for e in edges:
            vertices = e.split()
            graph[int(vertices[0])].append(int(vertices[1]))

    # task limit for preventing performance issues
    TASKS = int(sys.argv[1])
    if TASKS > len(graph):
        print "You have more work items than recommended!"
        sys.exit()
    
    # constructing list of neighbours
    vertex = np.dtype([("id", np.int32), ("size", np.int32), ("start", np.int32), ("prev", np.int32)])
    struct_arr = np.empty(len(graph), vertex)

    fill = 0
    num_nbrs = 0
    for v, nbrs in graph.items():
        struct_arr[fill]["id"] = v
        struct_arr[fill]["size"] = len(nbrs)
        struct_arr[fill]["start"] = num_nbrs
        struct_arr[fill]["prev"] = -1
        for nbr in nbrs:
            nbr_list.append(nbr)
            num_nbrs += 1
        fill += 1
    
    nbr_list = np.array(nbr_list)
    nbr_list = nbr_list.astype(np.int32)

    # bfs stuff
    frontier = np.full(len(graph), -1) # next in line to be traversed with arbitrary source
    tmp = np.full(len(graph), -1)      # temporary frontier
    goal = np.array([int(sys.argv[3])])# vertex to find
    vertices = np.array([len(graph)])  # number of vertices
    found = np.array([0])              # search result indicator

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
    frontier_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=frontier)
    tmp_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=tmp)
    goal_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=goal)
    vertices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertices)
    sources_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sources)
    found_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found)

    # kernel program
    prg = cl.Program(ctx, '''
        typedef struct {
            int id;
            int size;
            int start;
            int prev;
        } vertex;

        __kernel void bfs (__global vertex structs[], __global int nbr_list[], __global int* frontier, __global int* tmp, __global int goal[], __global int vertices[], __global int sources[], __global int* found) {
            int gid = get_global_id(0);
            frontier[gid] = sources[gid];
            int unv = 2;
            int fsize = 1;
            if (gid>=fsize)
                fsize = gid + 1;
            int prs = fsize;
            while (fsize != 0) {
                if (fsize < unv) { // top-down
                    for(int i=gid; i<fsize; i+=prs) {
                        int current = frontier[i];
                        for (int s=0; s<vertices[0]; s++) {
                            if (found[0] > 0)
                                return;
                            if (structs[s].id == current) {
                                for (int n=structs[s].start; n<structs[s].start + structs[s].size; n++)  {
                                    int nb = nbr_list[n];
                                    for (int sn=0; sn<vertices[0]; sn++) {
                                        if (structs[sn].id == nb && structs[sn].prev == -1) {
                                            structs[sn].prev = current;
                                            int spot = 0;
                                            while (tmp[spot] != -1) {
                                                spot++;
                                            }
                                            tmp[spot] = nb;
                                            if (nb == goal[0]) {
                                                found[0]++;
                                                return;
                                            }
                                            break;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
                else {   // bottom-up
                    for (int s=gid; s<vertices[0]; s+=prs) {
                        if (found[0] > 0)
                            return;
                        if (structs[s].prev == -1) {
                            for (int n=structs[s].start; n<structs[s].start + structs[s].size; n++) {
                                int nb = nbr_list[n];
                                for (int sn=0; sn<vertices[0]; sn++) {
                                    if (structs[sn].id == nb && structs[sn].prev != -1) {
                                        structs[s].prev = nb;
                                        int spot = 0;
                                        while (tmp[spot] != -1) {
                                            spot++;
                                        }
                                        tmp[spot] = nb;
                                        if (nb == goal[0]) {
                                            found[0]++;
                                            return;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                
                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
                
                int nsize = 0;
                for (int fr=gid; fr<vertices[0]; fr+=prs) {
                    frontier[fr] = tmp[fr];
                    if (tmp[fr] != -1)
                        nsize++;
                    else
                        tmp[fr] = -1;
                }
                fsize = nsize;

                unv = 0;
                for (int u=0; u<vertices[0]; u++) {
                    if (structs[u].prev == -1)
                        unv++;
                }
            }
        }
    ''').build()

    start = time.time()
    event = prg.bfs(queue, (TASKS,), (1,), dev_struct_arr.data, nbr_list_buf, frontier_buf, tmp_buf, goal_buf, vertices_buf, sources_buf, found_buf)
    event.wait()
    print time.time() - start

    # print path if found
    narr = dev_struct_arr.map_to_host(None, None, True, None)
    printdict = {}
    for st in narr:
        printdict[str(st[0])] = st[3]
    
    gv = goal[0]
    pathStr = ""
    while True:
        for s in sources:
            if (s==gv):
                pathStr = str(s) + pathStr
                print pathStr
                sys.exit()
        pathStr = " -> " + str(gv) + pathStr
        gv = printdict[str(gv)]