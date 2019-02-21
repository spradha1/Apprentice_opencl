import pyopencl as cl
import os

TASKS = 40

if __name__ == "__main__":

    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    os.environ['PYOPENCL_CTX'] = '1'

    # platform, device, context & queue setup
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    prg = cl.Program(ctx, '''
        __kernel void hello_world() {
            int global_id = get_global_id(0);
            printf("Hello from kernel #%d\\n", global_id);
        }
        ''').build()

    prg.hello_world(queue, (TASKS,), (1,))