import csdl_alpha as csdl
from csdl_alpha.backends.jax.jax_simulator import JaxSimulator
import numpy as np
import pytest


def test_block_newton_basic_solve_and_derivatives():
    """
    Test BlockNewton on a coupled 2-state nonlinear system.
    Verify both forward solve and total reverse AD derivatives.
    """
    rec = csdl.Recorder(inline=True)
    rec.start()

    a = csdl.Variable(value=1.5)
    b = csdl.Variable(value=2.0)
    c = csdl.Variable(value=-1.0)
    x = csdl.ImplicitVariable((1,), value=0.34)

    # Residual: r(x) = x - (-a*x^2 - c)/b = 0
    # Equivalent to a*x^2 + b*x + c = 0
    # Positive root: (-2 + sqrt(4 - 4*1.5*(-1))) / (2*1.5) = (-2 + sqrt(10)) / 3 ≈ 0.3874258867
    ax2 = a * x**2
    res = x - (-ax2 - c) / b

    solver = csdl.nonlinear_solvers.BlockNewton(print_status=False, tolerance=1e-12)
    solver.add_state(x, res)
    solver.run()

    obj = x**2
    rec.stop()

    sim = JaxSimulator(rec, additional_inputs=[a, b, c], additional_outputs=[x, obj])
    sim.run()

    expected_x = (-2.0 + np.sqrt(10.0)) / 3.0
    np.testing.assert_allclose(sim[x], [expected_x], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sim[obj], [expected_x**2], rtol=1e-7, atol=1e-7)

    # Verify total derivatives
    sim.check_totals()


def test_block_newton_kkt_system():
    """
    Test BlockNewton with add_kkt_constraint on an equality-constrained quadratic program:
        min  0.5 * x^T W x
        s.t. A x = b
    KKT conditions:
        r_x = W x + A^T lambda = 0
        r_lambda = A x - b = 0
    """
    rec = csdl.Recorder(inline=True)
    rec.start()

    W_val = np.array([[2.0, 0.5], [0.5, 3.0]])
    A_val = np.array([[1.0, 1.0]])
    b_val = np.array([3.0])

    x = csdl.ImplicitVariable((2,), value=np.array([0.0, 0.0]))
    lam = csdl.ImplicitVariable((1,), value=np.array([0.0]))

    W = csdl.Variable(value=W_val)
    A = csdl.Variable(value=A_val)
    b = csdl.Variable(value=b_val)

    rx = csdl.matvec(W, x) + csdl.matvec(A.T(), lam)
    rlam = csdl.matvec(A, x) - b

    solver = csdl.nonlinear_solvers.BlockNewton(print_status=False, tolerance=1e-12)
    solver.add_state(x, rx)
    solver.add_kkt_constraint(lam, rlam)
    solver.run()

    obj = 0.5 * csdl.sum(x * csdl.matvec(W, x))
    rec.stop()

    sim = JaxSimulator(rec, additional_inputs=[b], additional_outputs=[x, lam, obj])
    sim.run()

    # Analytical solution via KKT matrix inversion
    K = np.block([[W_val, A_val.T], [A_val, np.zeros((1, 1))]])
    rhs = np.array([0.0, 0.0, 3.0])
    sol = np.linalg.solve(K, rhs)
    expected_x = sol[:2]
    expected_lam = sol[2:]

    np.testing.assert_allclose(sim[x], expected_x, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sim[lam], expected_lam, rtol=1e-7, atol=1e-7)

    # Check total derivatives d(obj)/d(b) == -lambda
    totals = sim.compute_totals()
    np.testing.assert_allclose(totals[obj, b], [[-expected_lam[0]]], rtol=1e-6, atol=1e-6)
    sim.check_totals()


def test_block_newton_automatic_sparsity():
    """
    Test BlockNewton on two uncoupled subsystems:
        Subsystem 1: r1(x1) = x1^2 - 4 = 0  => x1 = 2
        Subsystem 2: r2(x2) = x2^3 - 27 = 0 => x2 = 3
    BlockNewton should automatically detect that J_12 = 0 and J_21 = 0
    via rustworkx graph reachability.
    """
    rec = csdl.Recorder(inline=True)
    rec.start()

    c1 = csdl.Variable(value=4.0)
    c2 = csdl.Variable(value=27.0)

    x1 = csdl.ImplicitVariable((1,), value=1.0)
    x2 = csdl.ImplicitVariable((1,), value=1.0)

    r1 = x1**2 - c1
    r2 = x2**3 - c2

    solver = csdl.nonlinear_solvers.BlockNewton(print_status=False, tolerance=1e-12)
    solver.add_state(x1, r1)
    solver.add_state(x2, r2)
    solver.run()

    obj = x1 + x2
    rec.stop()

    sim = JaxSimulator(rec, additional_inputs=[c1, c2], additional_outputs=[x1, x2, obj])
    sim.run()

    np.testing.assert_allclose(sim[x1], [2.0], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sim[x2], [3.0], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sim[obj], [5.0], rtol=1e-7, atol=1e-7)

    # Verify derivatives
    sim.check_totals()


def test_block_newton_symmetric_primal():
    """
    Test BlockNewton with symmetric_primal=True on a coupled 2-state system:
        min 0.5*x1^2 + x1*x2 + 1.5*x2^2 - c1*x1 - c2*x2
        r1 = x1 + x2 - c1 = 0
        r2 = x1 + 3*x2 - c2 = 0
    Here H = [[1, 1], [1, 3]]. J_21 is the transpose of J_12.
    BlockNewton should evaluate J_12 and transpose it to obtain J_21.
    """
    rec = csdl.Recorder(inline=True)
    rec.start()

    c1 = csdl.Variable(value=3.0)
    c2 = csdl.Variable(value=5.0)

    x1 = csdl.ImplicitVariable((1,), value=0.0)
    x2 = csdl.ImplicitVariable((1,), value=0.0)

    r1 = x1 + x2 - c1
    r2 = x1 + 3.0 * x2 - c2

    solver = csdl.nonlinear_solvers.BlockNewton(
        print_status=False, tolerance=1e-12, symmetric_primal=True
    )
    solver.add_state(x1, r1)
    solver.add_state(x2, r2)
    solver.run()

    obj = x1**2 + x2**2
    rec.stop()

    # Verify that the transpose block was registered
    assert (r2, x1) in solver._transpose_blocks
    assert solver._transpose_blocks[(r2, x1)] == (r1, x2, 1.0)

    sim = JaxSimulator(rec, additional_inputs=[c1, c2], additional_outputs=[x1, x2, obj])
    sim.run()

    # Solution: x1 + x2 = 3, x1 + 3*x2 = 5 => 2*x2 = 2 => x2 = 1, x1 = 2
    np.testing.assert_allclose(sim[x1], [2.0], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sim[x2], [1.0], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sim[obj], [5.0], rtol=1e-7, atol=1e-7)

    # Check total derivatives
    sim.check_totals()


def test_block_newton_constant_blocks():
    """
    Test BlockNewton with set_block_constant to bypass autodiff for known blocks.
    System:
        r(x) = A x - b = 0  => d(r)/d(x) = A (constant)
    """
    rec = csdl.Recorder(inline=True)
    rec.start()

    A_mat = np.array([[3.0, 1.0], [1.0, 2.0]])
    b_val = np.array([5.0, 4.0])

    x = csdl.ImplicitVariable((2,), value=np.array([0.0, 0.0]))
    A = csdl.Variable(value=A_mat)
    b = csdl.Variable(value=b_val)

    r = csdl.matvec(A, x) - b

    solver = csdl.nonlinear_solvers.BlockNewton(print_status=False, tolerance=1e-12)
    solver.add_state(x, r)
    # Supply exact analytical constant Jacobian block
    solver.set_block_constant(r, x, A_mat)
    solver.run()

    obj = csdl.sum(x**2)
    rec.stop()

    assert (r, x) in solver._constant_blocks

    sim = JaxSimulator(rec, additional_inputs=[b], additional_outputs=[x, obj])
    sim.run()

    expected_x = np.linalg.solve(A_mat, b_val)
    np.testing.assert_allclose(sim[x], expected_x, rtol=1e-7, atol=1e-7)
    sim.check_totals()


def test_block_newton_combined_kkt_and_primal_symmetry():
    """
    Test BlockNewton with both KKT constraint symmetry and primal Hessian symmetry:
        min  0.5*x1^2 + x1*x2 + x2^2
        s.t. x1 + 2*x2 = 4
    KKT conditions:
        r1 = x1 + x2 + lam = 0
        r2 = x1 + 2*x2 + 2*lam = 0
        rc = x1 + 2*x2 - 4 = 0
    Block structure:
        - J(rc, lam) = 0 (zero block)
        - J(r1, lam) = J(rc, x1)^T = 1 (KKT transpose)
        - J(r2, lam) = J(rc, x2)^T = 2 (KKT transpose)
        - J(r2, x1) = J(r1, x2)^T = 1 (Primal Hessian transpose)
    """
    rec = csdl.Recorder(inline=True)
    rec.start()

    x1 = csdl.ImplicitVariable((1,), value=0.0)
    x2 = csdl.ImplicitVariable((1,), value=0.0)
    lam = csdl.ImplicitVariable((1,), value=0.0)

    target_b = csdl.Variable(value=4.0)

    r1 = x1 + x2 + lam
    r2 = x1 + 2.0 * x2 + 2.0 * lam
    rc = x1 + 2.0 * x2 - target_b

    solver = csdl.nonlinear_solvers.BlockNewton(
        print_status=False, tolerance=1e-12, symmetric_primal=True
    )
    solver.add_state(x1, r1)
    solver.add_state(x2, r2)
    solver.add_kkt_constraint(lam, rc)
    solver.run()

    obj = 0.5 * x1**2 + x1 * x2 + x2**2
    rec.stop()

    # Verify blocks configured
    assert (rc, lam) in solver._zero_blocks
    assert (r1, lam) in solver._transpose_blocks
    assert (r2, lam) in solver._transpose_blocks
    assert (r2, x1) in solver._transpose_blocks

    sim = JaxSimulator(rec, additional_inputs=[target_b], additional_outputs=[x1, x2, lam, obj])
    sim.run()

    # Analytical solve of KKT system:
    # [1 1 1; 1 2 2; 1 2 0] [x1; x2; lam] = [0; 0; 4]
    K = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 2.0, 0.0]])
    sol = np.linalg.solve(K, np.array([0.0, 0.0, 4.0]))
    np.testing.assert_allclose(sim[x1], [sol[0]], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sim[x2], [sol[1]], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sim[lam], [sol[2]], rtol=1e-7, atol=1e-7)

    sim.check_totals()

