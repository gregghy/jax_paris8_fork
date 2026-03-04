import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import time

jax.config.update("jax_enable_x64", True)

def xoshiro128pp_kernel(s0_ref, s1_ref, s2_ref, s3_ref,
                        s0_out, s1_out, s2_out, s3_out,
                        out_ref):
    """
    RNG kernel, runs on a single thread.
    State in and out need to be sliced before running the kernel
    Pallas GPU doesn't support slicing at the moment
    """
    s0 = s0_ref[...]
    s1 = s1_ref[...]
    s2 = s2_ref[...]
    s3 = s3_ref[...]
    
    def rotl32(x, k):
        return (x << jnp.uint32(k)) | (x >> jnp.uint32(32 - k))

    result = rotl32(s0 + s3, 7) + s0
    
    t = s1 << jnp.uint32(9)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = rotl32(s3, 11)
   
    s0_out[...] = s0
    s1_out[...] = s1
    s2_out[...] = s2
    s3_out[...] = s3
    out_ref[...] = result

def splitmix32_seed_kernel(seed_ref, s0_ref, s1_ref, s2_ref, s3_ref):
    pid = pl.program_id(0)
    block_size = s0_ref.shape[0]
    
    global_seed = seed_ref[0].astype(jnp.uint32)
    offsets = jax.lax.iota(jnp.uint32, block_size)
    base_id = (jnp.uint32(pid) * jnp.uint32(block_size)) + offsets
    
    thread_state = base_id + global_seed

    # 32-bit Weyl constant (golden ratio)
    weyl_const = jnp.uint32(0x9E3779B9)

    def mix32(z):
        """MurmurHash3 32-bit mixing function"""
        z = (z ^ (z >> jnp.uint32(16))) * jnp.uint32(0x85ebca6b)
        z = (z ^ (z >> jnp.uint32(13))) * jnp.uint32(0xc2b2ae35)
        return z ^ (z >> jnp.uint32(16))

    s0_ref[...] = mix32(thread_state + weyl_const)
    s1_ref[...] = mix32(thread_state + weyl_const * jnp.uint32(2))
    s2_ref[...] = mix32(thread_state + weyl_const * jnp.uint32(3))
    s3_ref[...] = mix32(thread_state + weyl_const * jnp.uint32(4))

def splitmix64_seed_kernel(seed_ref, s0_ref, s1_ref, s2_ref, s3_ref):
    pid = pl.program_id(0)
    block_size = s0_ref.shape[0]
    global_seed = seed_ref[0]

    offsets = jax.lax.iota(jnp.uint64, block_size)
    base_id = (jnp.uint64(pid) * jnp.uint64(block_size)) + offsets
    thread_ids = base_id + global_seed

    c1 = (jnp.uint64(0x9e3779b9) << jnp.uint64(32)) | jnp.uint64(0x7f4a7c15)
    c2 = (jnp.uint64(0xbf58476d) << jnp.uint64(32)) | jnp.uint64(0x1ce4e5b9)
    c3 = (jnp.uint64(0x94d049bb) << jnp.uint64(32)) | jnp.uint64(0x133111eb)

    def next_splitmix(state):
        state = state + c1
        z = state
        z = (z ^ (z >> jnp.uint64(30))) * c2
        z = (z ^ (z >> jnp.uint64(27))) * c3
        return state, z ^ (z >> jnp.uint64(31))

    state, r1 = next_splitmix(thread_ids)
    _, r2 = next_splitmix(state)

    s0_ref[...] = (r1 >> jnp.uint64(32)).astype(jnp.uint32)
    s1_ref[...] = r1.astype(jnp.uint32)
    s2_ref[...] = (r2 >> jnp.uint64(32)).astype(jnp.uint32)
    s3_ref[...] = r2.astype(jnp.uint32)

def generate_random_batch(states, block_size):
    if isinstance(states, tuple) or isinstance(states, list):
        s0, s1, s2, s3 = states
        batch_size = s0.shape[0]
    else:
        batch_size = states.shape[0]
        s0, s1, s2, s3 = states[:, 0], states[:, 1], states[:, 2], states[:, 3]

    outputs = pl.pallas_call(
        xoshiro128pp_kernel,
        in_specs=[
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
        ],
        out_specs=[
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,))
        ],
        grid=(batch_size // block_size,),
        out_shape=[
            jax.ShapeDtypeStruct((batch_size,), jnp.uint32),
            jax.ShapeDtypeStruct((batch_size,), jnp.uint32),
            jax.ShapeDtypeStruct((batch_size,), jnp.uint32),
            jax.ShapeDtypeStruct((batch_size,), jnp.uint32),
            jax.ShapeDtypeStruct((batch_size,), jnp.uint32)
        ],
        interpret=True
    )(s0, s1, s2, s3)

    out_s0, out_s1, out_s2, out_s3, random_numbers = outputs

    updated_states = (out_s0, out_s1, out_s2, out_s3)
    return updated_states, random_numbers


def generate_seeds(total_threads, global_seed, block_size, bit_size=64):
    grid = (total_threads // block_size,)
    
    if (bit_size == 64):
        seed_arr = jnp.array([global_seed], dtype=jnp.uint64)
        s0, s1, s2, s3 = pl.pallas_call(
            splitmix64_seed_kernel,
            in_specs=[
                pl.BlockSpec((1,), lambda i: (0,))
            ],
            out_specs=[
                # Corrected: Just return 'i'
                pl.BlockSpec((block_size,), lambda i: (i,)),
                pl.BlockSpec((block_size,), lambda i: (i,)),
                pl.BlockSpec((block_size,), lambda i: (i,)),
                pl.BlockSpec((block_size,), lambda i: (i,))
            ],
            grid=grid,
            out_shape=[
                jax.ShapeDtypeStruct((total_threads,), jnp.uint32),
                jax.ShapeDtypeStruct((total_threads,), jnp.uint32),
                jax.ShapeDtypeStruct((total_threads,), jnp.uint32),
                jax.ShapeDtypeStruct((total_threads,), jnp.uint32)
            ],
            interpret=True
        )(seed_arr)
    else:
        seed_arr = jnp.array([global_seed], dtype=jnp.uint64)
        s0, s1, s2, s3 = pl.pallas_call(
            splitmix32_seed_kernel,
            in_specs=[
                pl.BlockSpec((1,), lambda i: (0,))
            ],
            out_specs=[
                # Corrected: Just return 'i'
                pl.BlockSpec((block_size,), lambda i: (i,)),
                pl.BlockSpec((block_size,), lambda i: (i,)),
                pl.BlockSpec((block_size,), lambda i: (i,)),
                pl.BlockSpec((block_size,), lambda i: (i,))
            ],
            grid=grid,
            out_shape=[
                jax.ShapeDtypeStruct((total_threads,), jnp.uint32),
                jax.ShapeDtypeStruct((total_threads,), jnp.uint32),
                jax.ShapeDtypeStruct((total_threads,), jnp.uint32),
                jax.ShapeDtypeStruct((total_threads,), jnp.uint32)
            ],
            interpret=True
        )(seed_arr)


    return (s0, s1, s2, s3)

if __name__ == "__main__":
    """
    total_threads = 2**27
    dynamic_seed = time.time_ns() # TODO hash this, maybe not necessary
    start_time_64 = time.time()
    states64 = generate_seeds(total_threads, global_seed=dynamic_seed, block_size=256, bit_size=64)
    end_time64 = time.time()

    start_time_32 = time.time()
    states32 = generate_seeds(total_threads, global_seed=dynamic_seed, block_size=256, bit_size=32)
    end_time32 = time.time()
  

    time64 = end_time64 - start_time_64
    time32 = end_time32 - start_time_32

    print(f"Execution time 64: {time64:.5f} seconds")
    print(f"Execution time 32: {time32:.5f} seconds")
    """
    dynamic_seed = time.time_ns() # TODO hash this, maybe not necessary
    states = generate_seeds(total_threads=2**8, global_seed=dynamic_seed, block_size=256, bit_size=64)
    print("--- Initial States for Xoshiro ---")
    print(states)

    updated_states, random_nums = generate_random_batch(states, block_size=256)

    print("\n--- Updated States ---")
    
    print("\n--- Random Numbers ---")
    print(random_nums)
