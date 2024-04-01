if __name__ == '__main__':

    import numpy as np

    x = np.ones((5,5))
    x[(1,3), (1,3)] = 0
    print(x)
    print()
    x = np.ones((5,5))
    x[1:4:2, 1:4:2] = 0
    print(x)
    print()
    x = np.ones((5,5))
    x[[(1,1), (3,3)]] = 0
    print(x)

    # x = x.set(y, csdl.slice[])
    # x = x.set(csdl.slice[], y)

    exit()

    import jax
    import jax.numpy as jnp
    import timeit
    import csdl

    x = jnp.arange(50).reshape(5,10)
    # x = jnp.arange(5)

    # option 1:
    # - jax api
    x = x.at[1:3, 5:].set(1)

    # option 2:
    # - slice might be misleading because we can provide values
    x = x.slice[1:3, 5:].set(1)

    # option 3:
    # - makes it obvious that it return a new variable
    x = csdl.set(x.slice[1:3, 5:], 1)
    
    x = x.slice[1:3, 5:].assign(1)

    # option 2:
    x = x.set(csdl.slice[1:3, 5:], 1)

    # option 3:
    x = x.set(slice(1,3), slice(5), 1)
    print(x)


    x = x.set(csdl.slice[1:3, 5:, (1,3,5)], 1)

    x = x.set(csdl.slice[(1,3), :], 1)

    

