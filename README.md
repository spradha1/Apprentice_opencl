# Apprentice_pyopencl
Basic examples of GPU programming via pyopencl

## Commands for specific examples

* [Breadth-First Search](###Breadth-FirstSearch)


### Breadth-First Search

Folder seq_vs_gpu has files performing breadth-first search on a file with edges with an arbitrary key to find. One may run the bfs_seq.py file with a file as the command line argument, whereas bfs_gpu would require number of work items and file arguments. One can look at the files in the dataSets folder to get familiar with the format.

`python bfs_seq.py <filename>`  
`python bfs_gpu <number of work_items> <filename>`