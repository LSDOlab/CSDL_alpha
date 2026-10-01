"""Ordered JAX callbacks for custom operations.

Under MPI, run this file directly to check that callbacks with collectives do
not overlap (test_mpi does that when mpirun is available):

    OMP_NUM_THREADS=1 mpirun -n 4 python test_ordered_callbacks.py reverse|forward [unordered] [concat]

``unordered`` restores the old behavior (wrong, rank-dependent results);
``concat`` adds derivatives_kwargs={'concatenate_ofs': True}.
"""
import os
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import csdl_alpha as csdl
from csdl_alpha.backends.jax.graph_to_jax import create_jax_function

jax.config.update("jax_enable_x64", True)   # as JaxSimulator does


class Square(csdl.CustomExplicitOperation):
    """y = x**2, counting its forward and reverse calls."""
    def __init__(self, ordered_callbacks=None):
        super().__init__()
        self.ordered_callbacks = ordered_callbacks
        self.calls = {"fwd": 0, "rev": 0}

    def evaluate(self, x):
        self.declare_input("x", x)
        y = self.create_output("y", x.shape)
        self.declare_derivative_parameters("y", "x")
        return y

    def compute(self, inputs, outputs):
        self.calls["fwd"] += 1
        outputs["y"] = inputs["x"] ** 2

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        self.calls["rev"] += 1
        d_inputs["x"] = 2.0 * inputs["x"] * d_outputs["y"]


class SquareImplicit(csdl.experimental.CustomImplicitOperation):
    """y with R = y - x**2 = 0."""
    def __init__(self, ordered_callbacks=None):
        super().__init__()
        self.ordered_callbacks = ordered_callbacks

    def evaluate(self, x):
        self.declare_input("x", x)
        return self.create_output("y", x.shape)

    def solve_residual_equations(self, inputs, outputs):
        outputs["y"] = inputs["x"] ** 2

    def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
        d_residuals["y"] = d_outputs["y"]

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        d_inputs["x"] = -2.0 * inputs["x"] * d_residuals["y"]


@pytest.mark.parametrize("op_class", [Square, SquareImplicit])
@pytest.mark.parametrize("ordered", [True, False, None])
def test_callbacks(op_class, ordered):
    """ordered_callbacks picks the forward and reverse callback (None: unordered
    without MPI); values and derivatives are unchanged."""
    x0, w = np.array([1.0, -2.0, 0.5]), np.array([1.0, 2.0, 3.0])
    rec = csdl.Recorder(inline=False)
    rec.start()
    x = csdl.Variable(name="x", value=x0)
    y = op_class(ordered).evaluate(x)
    f1, f2 = csdl.sum(y), csdl.sum(y * w)
    jaxpr = str(jax.make_jaxpr(create_jax_function(rec.get_root_graph(), [csdl.derivative(f1, x)], [x]))(x0))
    rec.stop()
    assert jaxpr.count("io_callback" if ordered else "pure_callback") == 2
    assert ("ordered=True" in jaxpr) == bool(ordered)

    sim = csdl.experimental.JaxSimulator(rec, gpu=False, additional_inputs=[x], additional_outputs=[f1, f2])
    sim.run()
    np.testing.assert_allclose(sim[f2], np.sum(w * x0 ** 2))
    totals = sim.compute_totals()
    np.testing.assert_allclose(totals[f1, x].ravel(), 2 * x0)
    np.testing.assert_allclose(totals[f2, x].ravel(), 2 * w * x0)


def test_unused_callbacks_do_not_run():
    """XLA must keep ordered callbacks, so the JAX function must leave out the
    operations the outputs do not need, such as the reverse pass that
    compute_totals records before run() is compiled."""
    rec = csdl.Recorder(inline=False)
    rec.start()
    x = csdl.Variable(name="x", value=np.array([1.0, 2.0]))
    used, unused = Square(True), Square(True)
    f = csdl.sum(used.evaluate(x))
    csdl.sum(unused.evaluate(x))
    rec.stop()
    sim = csdl.experimental.JaxSimulator(rec, gpu=False, additional_inputs=[x], additional_outputs=[f])
    sim.compute_totals()
    sim.run()
    assert used.calls == {"fwd": 2, "rev": 1}
    assert unused.calls == {"fwd": 0, "rev": 0}


def test_random_values_unchanged_by_skipped_operations():
    rec = csdl.Recorder(inline=False)
    rec.start()
    x = csdl.Variable(name="x", value=np.ones(3))
    a, b = csdl.normal((3,)) + x, csdl.normal((3,)) + x
    key = jax.random.PRNGKey(3)
    both = create_jax_function(rec.get_root_graph(), [a, b], [x])(jnp.ones(3), prng_key=key)
    only_b = create_jax_function(rec.get_root_graph(), [b], [x])(jnp.ones(3), prng_key=key)
    rec.stop()
    np.testing.assert_array_equal(both[1], only_b[0])


def test_vmap_falls_back_to_unordered():
    """Batched derivatives (loop=False) vmap the reverse pass."""
    x0 = np.array([1.0, -2.0, 0.5])
    rec = csdl.Recorder(inline=False)
    rec.start()
    x = csdl.Variable(name="x", value=x0)
    jac = csdl.derivative(Square(True).evaluate(x), x, loop=False)
    rec.stop()
    with pytest.warns(UserWarning, match="vmapped"):
        out = jax.jit(create_jax_function(rec.get_root_graph(), [jac], [x]))(jnp.asarray(x0))
    np.testing.assert_allclose(out[0], np.diag(2 * x0))


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
@pytest.mark.parametrize("pattern", ["reverse", "forward"])
def test_mpi(pattern):
    pytest.importorskip("mpi4py")
    result = subprocess.run(["mpirun", "-n", "4", sys.executable, os.path.abspath(__file__), pattern],
                            env=dict(os.environ, OMP_NUM_THREADS="1"), capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


def mpi_check(pattern, unordered, concat):
    """16 independent callbacks with an Allreduce each: one operation y = (sum_r A_r) x
    with 16 objectives (reverse), or 16 such operations (forward). Exits with 1 if
    any rank's values or gradients differ from the exact ones."""
    from mpi4py import MPI
    import scipy.sparse as sp
    comm, M, K = MPI.COMM_WORLD, 400, 16

    class DistributedMatvec(csdl.CustomExplicitOperation):
        ordered_callbacks = False if unordered else None

        def __init__(self, A):
            super().__init__()
            self.A = A

        def evaluate(self, x):
            self.declare_input("x", x)
            y = self.create_output("y", (M,))
            self.declare_derivative_parameters("y", "x")
            return y

        def compute(self, inputs, outputs):
            outputs["y"] = np.empty(M)
            comm.Allreduce(self.A @ inputs["x"], outputs["y"])

        def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
            d_inputs["x"] = np.empty(M)
            comm.Allreduce(np.ascontiguousarray(self.A.T @ d_outputs["y"]), d_inputs["x"])

    # rank-local sparse blocks, and the exact global matrices (no collectives needed later)
    A = [sp.random(M, M, density=0.01 * (comm.rank + 1), format="csr",
                   random_state=np.random.default_rng(100 * k + comm.rank))
         for k in range(1 if pattern == "reverse" else K)]
    A_global = [sum(comm.allgather(a)) for a in A]
    W = np.random.default_rng(7).standard_normal((K, M))
    x0 = np.sin(0.37 * np.arange(M)) + 0.1

    rec = csdl.Recorder(inline=False)
    rec.start()
    x = csdl.Variable(name="x", value=x0)
    if pattern == "reverse":
        y = DistributedMatvec(A[0]).evaluate(x)
        fs = [csdl.sum(y * W[k]) for k in range(K)]
        exact = [(W[k] @ A_global[0] @ x0, A_global[0].T @ W[k]) for k in range(K)]
    else:
        fs = [csdl.sum(DistributedMatvec(A[k]).evaluate(x) * W[0]) for k in range(K)]
        exact = [(W[0] @ A_global[k] @ x0, A_global[k].T @ W[0]) for k in range(K)]
    rec.stop()

    sim = csdl.experimental.JaxSimulator(rec, gpu=False, additional_inputs=[x], additional_outputs=fs,
                                         derivatives_kwargs={"concatenate_ofs": True} if concat else None)
    sim.run()
    totals = sim.compute_totals()
    err = max(max(abs(sim[f].item() - f_exact) / abs(f_exact),
                  np.linalg.norm(totals[f, x].ravel() - g_exact) / np.linalg.norm(g_exact))
              for f, (f_exact, g_exact) in zip(fs, exact))
    err = comm.allreduce(err, op=MPI.MAX)
    if comm.rank == 0:
        print(f"{pattern}, unordered={unordered}, concat={concat}: max relative error {err:.1e}", flush=True)
    sys.exit(int(err > 1e-10))


if __name__ == "__main__":
    mpi_check(sys.argv[1], "unordered" in sys.argv, "concat" in sys.argv)
