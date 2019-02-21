# square elements in an array
import pyopencl as cl
import numpy as np
import os

SIZE = 10

if __name__ == "__main__":
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    os.environ['PYOPENCL_CTX'] = '1'

    # platform, device, context & queue setup
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    # array attributes
    a = np.random.randint(10, size=SIZE)
    a = a.astype(np.int32)
    copy_a = a

    # buffer allocation
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

    # kernel program
    prg = cl.Program(ctx, '''
        __kernel void square(__global int a[]){
            int gid = get_global_id(0);
            a[gid] *= a[gid];
        }
        ''').build()

    prg.square(queue, (SIZE,), (1,), a_buf)
    a = np.empty((SIZE,), dtype = np.uint32)

    cl.enqueue_copy(queue, a, a_buf)

    print "array:"
    print copy_a
    print "squared array:"
    print a