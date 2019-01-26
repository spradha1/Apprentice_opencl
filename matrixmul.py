# multiply matrices of size n*m & m*p
import pyopencl as cl
import numpy as np
import os

if __name__ == "__main__":
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    os.environ['PYOPENCL_CTX'] = '1'

    # platform, device, context & queue setup
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    # matrices
    (n, m, p) = (1, 2, 3)
    a = np.random.randint(5, size=(n*m))
    b = np.random.randint(5, size=(m*p))
    c = np.zeros((n*p), dtype=np.int32)

    a = a.astype(np.int32)
    b = b.astype(np.int32)

    # buffer allocation
    mem_flags = cl.mem_flags
    a_buf = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(ctx, mem_flags.WRITE_ONLY, c.nbytes)

    # kernel program
    prg = cl.Program(ctx, """
        __kernel void multiply(ushort n, ushort m, ushort p, __global int *a, __global int *b, __global int *c){
            int gid = get_global_id(0);
            c[gid] = 0.0f;
            int rowC = gid/p;
            int colC = gid%p;
            __global int *pA = &a[rowC*m];
            __global int *pB = &b[colC];
            for(int k=0; k<m; k++){
                pB = &b[colC+k*p];
                c[gid] += (*(pA++))*(*pB);
            }
        }
        """).build()

    prg.multiply(queue, c.shape, None,
                np.uint16(n), np.uint16(m), np.uint16(p),
                a_buf, b_buf, c_buf)

    cl.enqueue_copy(queue, c, c_buf)

    print "matrix A:"
    print a.reshape(n, m)
    print "matrix B:"
    print b.reshape(m, p)
    print "multiplied A*B:"
    print c.reshape(n, p)