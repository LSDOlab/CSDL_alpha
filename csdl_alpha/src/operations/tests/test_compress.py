"""Option, error, and exact-match tests for csdl.experimental.compress.

Each test compares a compressed model with the same model left uncompressed.
The backend-matrix tests (values and finite-difference derivatives) are
TestJaxCompress in compress_operations.py.
"""
import numpy as np
import pytest

import csdl_alpha as csdl
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
from csdl_alpha.experimental import JaxCompressedOperation, compress


def model(x, y, c):
    """Region to compress: a few elementwise ops, a matvec, and a reduction."""
    a = csdl.sin(x) * y + csdl.exp(0.3 * x)
    A = csdl.reshape(a, (2, 2))
    b = csdl.matvec(A, c)
    z = csdl.sum(b**2) + csdl.norm(a)
    return a, z


def build(do_compress, inline=True, compress_outputs=('z',), **compress_kwargs):
    rec = csdl.Recorder(inline=inline)
    rec.start()
    x = csdl.Variable(name='x', value=np.array([0.1, 0.5, -0.3, 1.2]))
    y = csdl.Variable(name='y', value=np.array([1.0, 2.0, 0.5, -1.0]))
    c = csdl.Variable(name='c', value=np.array([0.7, -0.2]))  # not passed to compress
    a, z = model(x, y, c)
    named = {'a': a, 'z': z}
    op = compress([x, y], [named[k] for k in compress_outputs], **compress_kwargs) if do_compress else None
    # downstream of the region, outside it
    f = z * 2.0 + csdl.sum(x)
    return rec, dict(x=x, y=y, c=c, a=a, z=z, f=f, op=op)


def op_names(rec):
    from csdl_alpha.src.graph.operation import Operation
    return sorted(n.name for n in rec.active_graph.node_table if isinstance(n, Operation))


def test_feeders_stay_outside_by_default():
    ref_rec, ref = build(False)
    ref_d = csdl.derivative(ref['f'], [ref['x'], ref['c']])
    ref_rec.stop()

    rec, v = build(True)
    d = csdl.derivative(v['f'], [v['x'], v['c']])
    rec.stop()
    names = [n for n in op_names(rec) if not n.startswith('vjp')]
    assert names.count('reshape') == 2
    assert v['c'] not in v['op'].inputs
    for k in ['x', 'c']:
        np.testing.assert_allclose(d[v[k]].value, ref_d[ref[k]].value, rtol=1e-12)


@pytest.mark.parametrize('loop', [True, False])
def test_values_and_first_derivatives(loop):
    ref_rec, ref = build(False)
    ref_d = csdl.derivative(ref['f'], [ref['x'], ref['y'], ref['c']], loop=loop)
    ref_rec.stop()

    rec, v = build(True)
    d = csdl.derivative(v['f'], [v['x'], v['y'], v['c']], loop=loop)
    rec.stop()

    np.testing.assert_allclose(v['f'].value, ref['f'].value)
    for k in ['x', 'y', 'c']:
        np.testing.assert_allclose(d[v[k]].value, ref_d[ref[k]].value, rtol=1e-12)


def test_absorbed_feeders_make_region_one_node():
    rec, v = build(True, absorb_feeders=True)
    rec.stop()
    op = v['op']
    assert isinstance(op, JaxCompressedOperation)
    # The reshapes of `c` and of the literal 0.3 feed only the region, so they are
    # absorbed; the literal is compiled in. Left outside: the downstream ops.
    assert op_names(rec) == ['add', 'jax_compressed', 'mult', 'sum']
    assert op.inputs == [v['x'], v['y'], v['c']]
    assert op.extra_inputs == [v['c']]
    assert op.outputs == [v['z']]
    assert v['a'] not in rec.active_graph.node_table


def test_second_derivatives():
    ref_rec, ref = build(False)
    g = csdl.derivative(ref['f'], ref['x'])
    H_ref = csdl.derivative(g, [ref['x'], ref['c']])
    ref_rec.stop()

    rec, v = build(True)
    g = csdl.derivative(v['f'], v['x'])
    H = csdl.derivative(g, [v['x'], v['c']])
    rec.stop()

    np.testing.assert_allclose(H[v['x']].value, H_ref[ref['x']].value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(H[v['c']].value, H_ref[ref['c']].value, rtol=1e-10, atol=1e-12)


def test_multiple_outputs_including_reused_intermediate():
    # `a` is both an output and an input to the computation of `z`.
    ref_rec, ref = build(False)
    ref_g = csdl.derivative(ref['f'] + csdl.sum(ref['a']**3), ref['x'])
    ref_rec.stop()

    rec, v = build(True, compress_outputs=('a', 'z'))
    g = csdl.derivative(v['f'] + csdl.sum(v['a']**3), v['x'])
    rec.stop()

    assert v['op'].outputs == [v['a'], v['z']]
    np.testing.assert_allclose(g.value, ref_g.value, rtol=1e-12)


def test_pysimulator_matches_finite_difference():
    rec, v = build(True, inline=False)
    rec.stop()
    sim = csdl.experimental.PySimulator(rec)
    sim[v['x']] = np.array([0.3, -0.1, 0.9, 0.2])
    sim.run()

    ref_rec, ref = build(False, inline=False)
    ref_rec.stop()
    ref_sim = csdl.experimental.PySimulator(ref_rec)
    ref_sim[ref['x']] = np.array([0.3, -0.1, 0.9, 0.2])
    ref_sim.run()
    np.testing.assert_allclose(sim[v['f']], ref_sim[ref['f']])

    wrts = [v['x'], v['y'], v['c']]
    exact = sim.compute_totals(v['f'], wrts)
    fd = sim.compute_totals(v['f'], wrts, use_finite_difference=True)
    for w in wrts:
        np.testing.assert_allclose(exact[v['f'], w], fd[v['f'], w], rtol=1e-5, atol=1e-6)


def test_jax_simulator():
    rec, v = build(True, inline=False)
    rec.stop()
    sim = csdl.experimental.JaxSimulator(
        rec, gpu=False, additional_inputs=[v['x'], v['c']], additional_outputs=[v['f']])
    sim[v['x']] = np.array([0.3, -0.1, 0.9, 0.2])
    sim.run()

    ref_rec, ref = build(False, inline=False)
    ref_rec.stop()
    ref_sim = csdl.experimental.JaxSimulator(
        ref_rec, gpu=False, additional_inputs=[ref['x'], ref['c']], additional_outputs=[ref['f']])
    ref_sim[ref['x']] = np.array([0.3, -0.1, 0.9, 0.2])
    ref_sim.run()
    np.testing.assert_allclose(sim[v['f']], ref_sim[ref['f']])

    d = sim.compute_totals()
    d_ref = ref_sim.compute_totals()
    for k in ['x', 'c']:
        np.testing.assert_allclose(d[v['f'], v[k]], d_ref[ref['f'], ref[k]], rtol=1e-12)


def test_nested_compress():
    ref_rec, ref = build(False)
    ref_d = csdl.derivative(ref['f'], [ref['x'], ref['c']])
    ref_rec.stop()

    rec, v = build(True)       # inner compress: x, y -> z
    outer = compress([v['x'], v['c']], v['f'], name='outer')
    d = csdl.derivative(v['f'], [v['x'], v['c']])
    rec.stop()

    assert 'jax_compressed' not in op_names(rec)  # the inner op was absorbed into `outer`
    assert outer.inputs[:2] == [v['x'], v['c']] and v['y'] in outer.inputs
    np.testing.assert_allclose(d[v['x']].value, ref_d[ref['x']].value, rtol=1e-12)
    np.testing.assert_allclose(d[v['c']].value, ref_d[ref['c']].value, rtol=1e-12)


def test_parameter_change_flows_through_extra_input():
    rec, v = build(True, inline=False)
    rec.stop()
    sim = csdl.experimental.PySimulator(rec)
    sim.run()
    f0 = sim[v['f']].copy()
    sim[v['c']] = np.array([3.0, 1.0])  # c was never passed to compress
    sim.run()
    assert not np.allclose(sim[v['f']], f0)


def test_loop_inside_region():
    def body(x, use_loop):
        if use_loop:
            acc = csdl.Variable(value=np.zeros(x.shape))
            for i in csdl.frange(3):
                acc = acc + csdl.sin(x * (i + 1.0))
        else:
            acc = sum(csdl.sin(x * float(i + 1)) for i in range(3))
        return csdl.sum(acc * x)

    results = []
    for do_compress in [False, True]:
        rec = csdl.Recorder(inline=True)
        rec.start()
        x = csdl.Variable(name='x', value=np.array([0.2, 0.4, 0.8]))
        z = body(x, use_loop=True)
        if do_compress:
            compress(x, z)
        g = csdl.derivative(z, x)
        rec.stop()
        results.append((z.value, g.value))
    np.testing.assert_allclose(results[1][0], results[0][0])
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=1e-12)


def test_escaping_intermediate_raises():
    rec = csdl.Recorder(inline=True)
    rec.start()
    x = csdl.Variable(value=np.ones(3))
    a = csdl.sin(x)
    z = csdl.sum(a)
    w = a * 2.0  # uses the intermediate `a` outside x -> z
    with pytest.raises(ValueError, match="intermediates='used'"):
        compress(x, z)
    rec.stop()


@pytest.mark.parametrize('intermediates', ['used', 'all'])
def test_intermediates_become_outputs(intermediates):
    results = []
    for do_compress in [False, True]:
        rec = csdl.Recorder(inline=True)
        rec.start()
        x = csdl.Variable(value=np.array([0.2, 0.7, 1.1]))
        a = csdl.sin(x)
        b = csdl.exp(a)
        z = csdl.sum(b * x)
        w = csdl.sum(a * 2.0)  # uses the intermediate `a` outside x -> z
        if do_compress:
            op = compress(x, z, intermediates=intermediates)
        g = csdl.derivative(z + w, x)
        rec.stop()
        results.append((w.value, g.value))
    if intermediates == 'used':
        assert op.extra_outputs == [a]
    else:
        assert set(op.extra_outputs) >= {a, b} and z not in op.extra_outputs
    assert op.outputs[0] is z
    np.testing.assert_allclose(results[1][0], results[0][0])
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=1e-12)


def test_device_and_jit_kwargs_reach_jit_and_derivatives():
    import jax
    cpu = jax.devices('cpu')[0]
    ref_rec, ref = build(False)
    ref_g = csdl.derivative(ref['f'], ref['x'])
    ref_rec.stop()

    rec, v = build(True, device='cpu', jit_kwargs={'keep_unused': True})
    g = csdl.derivative(v['f'], v['x'], loop=False)
    rec.stop()
    np.testing.assert_allclose(g.value, ref_g.value, rtol=1e-12)
    from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation as _S
    vjp_ops = []
    stack = [rec.active_graph]
    while stack:
        for n in stack.pop().node_table:
            if isinstance(n, JaxCompressedOperation) and n.name.startswith('vjp'):
                vjp_ops.append(n)
            elif isinstance(n, _S):
                stack.append(n.get_subgraph())
    for op in [v['op']] + vjp_ops:
        assert op.device == cpu and op.jit_kwargs == {'keep_unused': True}
    assert vjp_ops


def test_bad_options_raise_before_touching_graph():
    rec = csdl.Recorder(inline=True)
    rec.start()
    x = csdl.Variable(value=np.ones(3))
    z = csdl.sum(csdl.sin(x))
    before = op_names(rec)
    with pytest.raises(ValueError, match='No JAX device'):
        compress(x, z, device='tpu')
    with pytest.raises(ValueError, match='intermediates must be'):
        compress(x, z, intermediates='some')
    assert op_names(rec) == before
    rec.stop()


def test_output_independent_of_inputs_raises():
    rec = csdl.Recorder(inline=True)
    rec.start()
    x = csdl.Variable(value=np.ones(3))
    y = csdl.Variable(value=np.ones(3))
    z = csdl.sum(y)
    with pytest.raises(ValueError, match='does not affect'):  # CSDL's own check
        compress(x, z)
    rec.stop()


def test_input_depending_on_input_raises():
    rec = csdl.Recorder(inline=True)
    rec.start()
    x = csdl.Variable(value=np.ones(3))
    y = csdl.sin(x)
    z = csdl.sum(x * y)
    with pytest.raises(ValueError, match='depends on another input'):
        compress([x, y], z)
    rec.stop()


def test_derivative_is_one_compiled_node():
    from csdl_alpha.src.graph.operation import Operation

    def visible_ops(graph):
        """Operations in `graph` and nested subgraphs, not looking inside JaxCompressedOperations."""
        for node in graph.node_table:
            if isinstance(node, Operation):
                yield node
                if isinstance(node, SubgraphOperation) and not isinstance(node, JaxCompressedOperation):
                    yield from visible_ops(node.get_subgraph())

    rec, v = build(True)
    csdl.derivative(v['f'], v['x'])
    rec.stop()
    names = [op.name for op in visible_ops(rec.active_graph)]
    assert names.count('vjp_jax_compressed') == 1
    assert not {'sin', 'exp', 'matvec', 'norm'} & set(names)  # region and its VJP stay inside
    assert isinstance(v['op'], SubgraphOperation)
    assert 'sin' in {op.name for op in visible_ops(v['op'].get_subgraph())}


def _implicit_model():
    a = csdl.Variable(name='a', value=np.array([2.0, 3.0]))
    x = csdl.ImplicitVariable(name='x', value=np.ones(2))
    solver = csdl.nonlinear_solvers.Newton(print_status=False, tolerance=1e-12)
    solver.add_state(x, x**3 + a * x - 10.0)
    solver.run()
    return a, csdl.sum(x**2 * a)


def test_implicit_operation_inside_region():
    # CSDL's implicit-function-theorem derivative is compiled into the VJP op.
    results = []
    for do_compress in [False, True]:
        rec = csdl.Recorder(inline=True)
        rec.start()
        a, z = _implicit_model()
        if do_compress:
            compress(a, z)
        g = csdl.derivative(z, a)
        H = csdl.derivative(g, a)
        rec.stop()
        results.append((z.value, g.value, H.value))
    for ref, got in zip(*results):
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_implicit_operation_in_jax_simulator():
    totals = []
    for do_compress in [False, True]:
        rec = csdl.Recorder(inline=False)
        rec.start()
        a, z = _implicit_model()
        if do_compress:
            compress(a, z)
        rec.stop()
        sim = csdl.experimental.JaxSimulator(rec, gpu=False, additional_inputs=[a], additional_outputs=[z])
        sim[a] = np.array([1.5, 4.0])
        sim.run()
        totals.append((sim[z], sim.compute_totals()[z, a]))
    np.testing.assert_allclose(totals[1][0], totals[0][0], rtol=1e-12)
    np.testing.assert_allclose(totals[1][1], totals[0][1], rtol=1e-10)


class CubePlusProduct(csdl.CustomExplicitOperation):
    """f = x**3 + x*y, elementwise, with hand-written derivatives."""

    def evaluate(self, x, y):
        self.declare_input('x', x)
        self.declare_input('y', y)
        f = self.create_output('f', x.shape)
        self.declare_derivative_parameters('f', 'x')
        self.declare_derivative_parameters('f', 'y')
        return f

    def compute(self, input_vals, output_vals):
        output_vals['f'] = input_vals['x']**3 + input_vals['x'] * input_vals['y']

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        derivatives['f', 'x'] = np.diag(3 * input_vals['x']**2 + input_vals['y'])
        derivatives['f', 'y'] = np.diag(input_vals['x'])


def _custom_model(x, y):
    u = csdl.sin(x) * 2.0
    f = CubePlusProduct().evaluate(u, y)
    return csdl.sum(csdl.exp(0.1 * f) * u)


def _build_custom(do_compress, inline, **compress_kwargs):
    rec = csdl.Recorder(inline=inline)
    rec.start()
    x = csdl.Variable(name='x', value=np.array([0.3, -0.4, 0.9]))
    y = csdl.Variable(name='y', value=np.array([1.0, 0.5, -2.0]))
    z = _custom_model(x, y)
    op = compress([x, y], z, **compress_kwargs) if do_compress else None
    f = 3.0 * z
    return rec, x, y, f, op


@pytest.mark.parametrize('loop', [True, False])
def test_custom_operation_uses_csdl_derivatives(loop):
    ref_rec, rx_, ry_, rf, _ = _build_custom(False, True)
    ref_d = csdl.derivative(rf, [rx_, ry_], loop=loop)
    ref_rec.stop()

    rec, x, y, f, op = _build_custom(True, True)
    d = csdl.derivative(f, [x, y], loop=loop)
    rec.stop()

    np.testing.assert_allclose(f.value, rf.value)
    np.testing.assert_allclose(d[x].value, ref_d[rx_].value, rtol=1e-12)
    np.testing.assert_allclose(d[y].value, ref_d[ry_].value, rtol=1e-12)


def test_custom_operation_in_simulators():
    ref_rec, rx_, ry_, rf, _ = _build_custom(False, False)
    ref_rec.stop()
    ref_sim = csdl.experimental.PySimulator(ref_rec)
    ref_sim.run()
    ref_d = ref_sim.compute_totals(rf, [rx_, ry_])

    rec, x, y, f, _ = _build_custom(True, False)
    rec.stop()
    for sim in [csdl.experimental.PySimulator(rec),
                csdl.experimental.JaxSimulator(rec, gpu=False, additional_inputs=[x, y], additional_outputs=[f])]:
        sim.run()
        np.testing.assert_allclose(sim[f], ref_sim[rf])
        d = sim.compute_totals(f, [x, y]) if isinstance(sim, csdl.experimental.PySimulator) else sim.compute_totals()
        np.testing.assert_allclose(d[f, x], ref_d[rf, rx_], rtol=1e-12)
        np.testing.assert_allclose(d[f, y], ref_d[rf, ry_], rtol=1e-12)


def _jax_sim_results(rec, x, c, f, loop):
    sim = csdl.experimental.JaxSimulator(
        rec, gpu=False, additional_inputs=[x, c], additional_outputs=[f],
        derivatives_kwargs={'loop': loop})
    sim[x] = np.array([0.3, -0.1, 0.9, 0.2])
    sim.run()
    d = sim.compute_totals()
    return sim[f], d[f, x], d[f, c]


@pytest.mark.parametrize('compile_separately', [True, False])
@pytest.mark.parametrize('loop', [True, False])  # loop=False vmaps the VJP op
def test_both_compile_modes_under_jax_simulator(loop, compile_separately):
    ref_rec, ref = build(False, inline=False)
    ref_rec.stop()
    expected = _jax_sim_results(ref_rec, ref['x'], ref['c'], ref['f'], loop)

    rec, v = build(True, inline=False, compile_separately=compile_separately)
    rec.stop()
    got = _jax_sim_results(rec, v['x'], v['c'], v['f'], loop)
    for e, g in zip(expected, got):
        np.testing.assert_allclose(g, e, rtol=1e-12)


def test_compile_separately_keeps_region_out_of_enclosing_program():
    import jax
    from csdl_alpha.backends.jax.graph_to_jax import create_jax_function

    def primitives(compile_separately):
        rec, v = build(True, inline=False, compile_separately=compile_separately)
        rec.stop()
        fn = create_jax_function(rec.active_graph, [v['f']], [v['x'], v['y'], v['c']])
        jaxpr = jax.make_jaxpr(fn)(*(jax.numpy.asarray(v[k].value) for k in 'xyc'))
        return [eqn.primitive.name for eqn in jaxpr.eqns]

    assert 'pjit' in primitives(False) or 'jit' in primitives(False)
    separate = primitives(True)
    assert 'pure_callback' in separate and not {'pjit', 'jit'} & set(separate)


def test_compile_separately_nested_and_with_custom_and_implicit_ops():
    results = []
    for do_compress in [False, True]:
        rec = csdl.Recorder(inline=False)
        rec.start()
        a, z_implicit = _implicit_model()
        x = csdl.Variable(name='x', value=np.array([0.3, -0.4]))
        z_custom = _custom_model(x, a)
        if do_compress:
            compress([a, x], z_custom)                                   # inner, separate (default)
        f = z_implicit * z_custom
        if do_compress:
            compress([a, x], f, name='outer', compile_separately=False)  # outer, fused, contains the inner
        rec.stop()
        sim = csdl.experimental.JaxSimulator(rec, gpu=False, additional_inputs=[a, x], additional_outputs=[f])
        sim[a] = np.array([1.5, 4.0])
        sim.run()
        d = sim.compute_totals()
        results.append((sim[f], d[f, a], d[f, x]))
    for ref, got in zip(*results):
        np.testing.assert_allclose(got, ref, rtol=1e-10)


def _loop_model(compress_mode, loop_kind, use_index, inline):
    """Four iterations of h <- 0.9 tanh(sin(p h) + s h) + 0.1 h, optionally compressing the body's region."""
    rec = csdl.Recorder(inline=inline)
    rec.start()
    x = csdl.Variable(name='x', value=np.array([0.2, 0.5, 0.9]))
    p = csdl.Variable(name='p', value=np.array([1.3]))  # read from outside the loop

    def body(h, i):
        scale = (i + 1.0) * 0.3 if use_index else 0.3
        h_new = csdl.tanh(csdl.sin(h * p) + scale * h) * 0.9 + h * 0.1
        if compress_mode is not None:
            compress(h, h_new, compile_separately=(compress_mode == 'separate'))
        return h_new

    if loop_kind == 'frange':
        h = x
        for i in csdl.frange(4):
            h = body(h, i)
    else:
        with csdl.experimental.enter_loop(vals=[list(range(4))]) as loop_builder:
            i = loop_builder.get_loop_indices()
            h0 = loop_builder.initialize_feedback(x)
            h1 = body(h0, i)
            loop_builder.finalize_feedback(h0, h1)
        h = loop_builder.add_output(h1)
        loop_builder.finalize()
    return rec, x, p, csdl.sum(h**2)


@pytest.mark.parametrize('compress_mode', ['fused', 'separate'])
@pytest.mark.parametrize('use_index', [False, True])
@pytest.mark.parametrize('loop_kind', ['frange', 'enter_loop'])
def test_compress_inside_loop_body(loop_kind, use_index, compress_mode):
    from csdl_alpha.src.graph.operation import Operation

    def inline_results(mode):
        rec, x, p, f = _loop_model(mode, loop_kind, use_index, inline=True)
        results = [f.value]
        for loop in [True, False]:
            d = csdl.derivative(f, [x, p], loop=loop)
            results += [d[x].value, d[p].value]
        rec.stop()
        return rec, results

    def jax_sim_results(mode):
        rec, x, p, f = _loop_model(mode, loop_kind, use_index, inline=False)
        rec.stop()
        sim = csdl.experimental.JaxSimulator(rec, gpu=False, additional_inputs=[x, p], additional_outputs=[f])
        sim[x] = np.array([0.4, -0.3, 1.1])
        sim.run()
        d = sim.compute_totals()
        return [sim[f], d[f, x], d[f, p]]

    _, expected = inline_results(None)
    rec, got = inline_results(compress_mode)
    for g, e in zip(got, expected):
        np.testing.assert_allclose(g, e, rtol=1e-10, atol=1e-12)
    for g, e in zip(jax_sim_results(compress_mode), jax_sim_results(None)):
        np.testing.assert_allclose(g, e, rtol=1e-10, atol=1e-12)

    # The loop body holds one compressed operation in place of the sin/tanh chain.
    loops = [n for n in rec.active_graph.node_table
             if isinstance(n, SubgraphOperation) and not isinstance(n, JaxCompressedOperation)]
    body_names = [n.name for n in loops[0].get_subgraph().node_table if isinstance(n, Operation)]
    assert body_names.count('jax_compressed') == 1
    assert not {'sin', 'tanh'} & set(body_names)


# ---- Sharing compiled functions between identical regions ----------------------

def _blocks(n, exponent=2.0, scale=1.0, inline=True):
    """n copies of r = sum(sin(x * t) ** exponent * c) * scale, each with its own x, t; c is shared."""
    rec = csdl.Recorder(inline=inline)
    rec.start()
    c = csdl.Variable(name='c', value=np.array([0.5, -1.0, 2.0]))
    xs, ts, rs = [], [], []
    for k in range(n):
        x = csdl.Variable(name=f'x{k}', value=np.array([0.1, 0.4, 0.7]) + 0.2 * k)
        t = csdl.Variable(name=f't{k}', value=np.array([1.0 + 0.1 * k]))
        rs.append(csdl.sum(csdl.sin(x * t) ** exponent * c) * scale)
        xs.append(x)
        ts.append(t)
    return rec, c, xs, ts, rs


def _total(rs):
    return sum(rs[1:], rs[0])


def _block_results(n, compress_fn, wrt=lambda xs, ts, c: xs + ts + [c]):
    rec, c, xs, ts, rs = _blocks(n)
    ops = compress_fn(xs, ts, rs)
    f = _total(rs)
    d = csdl.derivative(f, wrt(xs, ts, c))
    rec.stop()
    return ops, rec, [f.value] + [d[w].value for w in wrt(xs, ts, c)]


def _assert_all_close(got, expected):
    for g, e in zip(got, expected):
        np.testing.assert_allclose(g, e, rtol=1e-10, atol=1e-12)


def test_identical_compress_calls_share_one_compiled_function():
    _, _, expected = _block_results(3, lambda xs, ts, rs: None)
    ops, rec, got = _block_results(3, lambda xs, ts, rs: [compress([x, t], r) for x, t, r in zip(xs, ts, rs)])
    _assert_all_close(got, expected)
    first, *others = ops
    assert first.shared_from is None
    assert all(op.shared_from is first for op in others)
    assert all(op.jit_fn is first.jit_fn for op in others)  # same input order: the very same jitted function
    rec.execute()                                            # evaluates all three forward operations
    assert first.jit_fn._cache_size() == 1                   # compiled once for all three

    # Their derivative operations share in the same way.
    vjps = _vjp_ops(rec)
    assert len(vjps) == 3 and sum(op.shared_from is None for op in vjps) == 1


def _all_nodes(graph):
    for node in graph.node_table:
        yield node
        if isinstance(node, SubgraphOperation) and not isinstance(node, JaxCompressedOperation):
            yield from _all_nodes(node.get_subgraph())


def test_sharing_with_inputs_in_a_different_order():
    _, _, expected = _block_results(2, lambda xs, ts, rs: None)
    ops, rec, got = _block_results(2, lambda xs, ts, rs: [compress([xs[0], ts[0]], rs[0]),
                                                          compress([ts[1], xs[1]], rs[1])])
    _assert_all_close(got, expected)
    assert ops[1].shared_from is ops[0] and ops[1]._input_permutation != [0, 1, 2]
    vjps = _vjp_ops(rec)  # derivative operations are built in the shared order, so they share too
    assert len(vjps) == 2 and sum(op.shared_from is None for op in vjps) == 1


@pytest.mark.parametrize('difference', ['exponent', 'constant'])
def test_regions_with_different_parameters_do_not_share(difference):
    rec = csdl.Recorder(inline=True)
    rec.start()
    x1 = csdl.Variable(value=np.array([0.3, 0.6]))
    x2 = csdl.Variable(value=np.array([0.3, 0.6]))
    if difference == 'exponent':        # a parameter stored on the operation
        r1, r2 = csdl.sum(csdl.sin(x1) ** 2.0), csdl.sum(csdl.sin(x2) ** 3.0)
    else:                               # a compiled-in literal constant
        r1, r2 = csdl.sum(csdl.sin(x1) * 2.0), csdl.sum(csdl.sin(x2) * 3.0)
    expected = r2.value.copy()
    op1, op2 = compress(x1, r1, absorb_feeders=True), compress(x2, r2, absorb_feeders=True)
    rec.stop()
    assert op2.shared_from is None
    sim = csdl.experimental.PySimulator(rec)
    sim.run()
    np.testing.assert_allclose(sim[r2], expected)


class _Scale(csdl.CustomExplicitOperation):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def evaluate(self, x):
        self.declare_input('x', x)
        f = self.create_output('f', x.shape)
        self.declare_derivative_parameters('f', 'x')
        return f

    def compute(self, input_vals, output_vals):
        output_vals['f'] = self.k * input_vals['x']**2

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        derivatives['f', 'x'] = np.diag(2 * self.k * input_vals['x'])


@pytest.mark.parametrize('assume', [False, True])
def test_custom_operations_share_only_when_assumed(assume):
    ks = [2.0, 2.0, 5.0]
    def build(do_compress):
        rec = csdl.Recorder(inline=True)
        rec.start()
        xs = [csdl.Variable(value=np.array([0.2, 0.9])) for _ in ks]
        rs = [csdl.sum(csdl.sin(_Scale(k).evaluate(x))) for k, x in zip(ks, xs)]
        # Assume only for the two that really match; the k=5 one never assumes.
        ops = [compress(x, r, assume_custom_ops_match=(assume and k == 2.0)) for k, x, r in zip(ks, xs, rs)] if do_compress else None
        f = _total(rs)
        g = csdl.derivative(f, xs)
        rec.stop()
        return ops, [f.value] + [g[x].value for x in xs]

    _, expected = build(False)
    ops, got = build(True)
    _assert_all_close(got, expected)
    assert (ops[1].shared_from is ops[0]) == assume
    assert ops[2].shared_from is None


def test_find_repeats_compresses_every_copy():
    _, _, expected = _block_results(3, lambda xs, ts, rs: None)
    ops, rec, got = _block_results(3, lambda xs, ts, rs: [compress([xs[0], ts[0]], rs[0], find_repeats=True)])
    _assert_all_close(got, expected)
    template = ops[0]
    assert len(template.repeats) == 2
    assert all(op.shared_from is template for op in template.repeats)
    names = [n.name for n in rec.active_graph.node_table if hasattr(n, 'inputs')]
    assert names.count('jax_compressed') == 3 and 'sin' not in names


def test_find_repeats_in_a_chain_takes_non_overlapping_copies():
    def build(do_compress, layers_per_copy):
        rec = csdl.Recorder(inline=True)
        rec.start()
        x = csdl.Variable(value=np.array([0.2, 0.5]))
        hs, h = [], x
        for _ in range(6):
            h = csdl.tanh(h * 0.9) + 0.05
            hs.append(h)
        op = compress(x, hs[layers_per_copy - 1], find_repeats=True) if do_compress else None
        f = csdl.sum(h**2)
        g = csdl.derivative(f, x)
        rec.stop()
        return op, [f.value, g.value]

    _, expected = build(False, 1)
    for layers_per_copy, num_repeats in [(1, 5), (2, 2), (4, 0)]:
        op, got = build(True, layers_per_copy)
        _assert_all_close(got, expected)
        assert len(op.repeats) == num_repeats


def test_find_repeats_skips_a_copy_with_an_escaping_intermediate():
    rec, c, xs, ts, rs = _blocks(3)
    # In block 1, the intermediate x1 * t1 is also used outside the block.
    leak = [n for n in rec.active_graph.node_table if hasattr(n, 'inputs') and xs[1] in n.inputs][0].outputs[0]
    w = csdl.sum(leak)
    op = compress([xs[0], ts[0]], rs[0], find_repeats=True)
    rec.stop()
    assert len(op.repeats) == 1 and op.repeats[0].outputs == [rs[2]]


def test_find_repeats_with_outputs_on_independent_paths():
    # Output 1 is not upstream of output 0, so its operations are found by
    # walking forward from the shared input.
    def build(do_compress):
        rec = csdl.Recorder(inline=True)
        rec.start()
        xs = [csdl.Variable(value=np.array([0.3, 0.8]) + k) for k in range(3)]
        outs = [(csdl.sum(csdl.sin(x)), csdl.cos(x) * 2.0) for x in xs]
        op = compress(xs[0], list(outs[0]), find_repeats=True) if do_compress else None
        f = sum((r + csdl.sum(q**2) for r, q in outs[1:]), outs[0][0] + csdl.sum(outs[0][1]**2))
        g = csdl.derivative(f, xs)
        rec.stop()
        return op, [f.value] + [g[x].value for x in xs]

    _, expected = build(False)
    op, got = build(True)
    _assert_all_close(got, expected)
    assert len(op.repeats) == 2


@pytest.mark.parametrize('compile_separately', [True, False])
def test_find_repeats_under_jax_simulator(compile_separately):
    def results(do_compress):
        rec, c, xs, ts, rs = _blocks(3, inline=False)
        op = compress([xs[0], ts[0]], rs[0], find_repeats=True, compile_separately=compile_separately) if do_compress else None
        f = _total(rs)
        rec.stop()
        sim = csdl.experimental.JaxSimulator(rec, gpu=False, additional_inputs=xs + ts + [c], additional_outputs=[f])
        sim.run()
        d = sim.compute_totals()
        return op, [sim[f]] + [d[f, w] for w in xs + ts + [c]]

    _, expected = results(False)
    op, got = results(True)
    _assert_all_close(got, expected)
    assert len(op.repeats) == 2
    if compile_separately:  # all three copies called one compiled function
        assert op.jit_fn._cache_size() == 1


# ---- Explicit sharing: compress(..., share_with=op) ------------------------------

def _vjp_ops(rec):
    # An operation can sit in more than one graph (a derivative loop's body reuses it), so dedupe.
    nodes = dict.fromkeys(_all_nodes(rec.active_graph))
    return [n for n in nodes if isinstance(n, JaxCompressedOperation) and n.name.startswith('vjp')]


def test_share_with_reuses_function_and_derivatives():
    _, _, expected = _block_results(2, lambda xs, ts, rs: None)
    def compress_both(xs, ts, rs):
        first = compress([xs[0], ts[0]], rs[0], share_compiled=False)
        second = compress([ts[1], xs[1]], rs[1], share_compiled=False, share_with=first)
        return [first, second]
    ops, rec, got = _block_results(2, compress_both)
    _assert_all_close(got, expected)
    assert ops[1].shared_from is ops[0]
    vjps = _vjp_ops(rec)
    assert len(vjps) == 2 and sum(op.shared_from is None for op in vjps) == 1


def test_share_with_second_derivatives_share_too():
    def build(do_compress):
        rec, c, xs, ts, rs = _blocks(2)
        if do_compress:
            first = compress([xs[0], ts[0]], rs[0], share_compiled=False)
            compress([xs[1], ts[1]], rs[1], share_compiled=False, share_with=first)
        f = _total(rs)
        g = csdl.derivative(f, xs)
        h = csdl.derivative(g[xs[0]] + g[xs[1]], xs)
        rec.stop()
        return rec, [h[x].value for x in xs]

    _, expected = build(False)
    rec, got = build(True)
    _assert_all_close(got, expected)
    vjps = _vjp_ops(rec)
    first_order = [op for op in vjps if not op.name.startswith('vjp_vjp')]
    second_order = [op for op in vjps if op.name.startswith('vjp_vjp')]
    assert len(first_order) == 2 and sum(op.shared_from is None for op in first_order) == 1
    assert len(second_order) == 2 and sum(op.shared_from is None for op in second_order) == 1


def test_share_with_custom_operations():
    def build(do_compress):
        rec = csdl.Recorder(inline=True)
        rec.start()
        xs = [csdl.Variable(value=np.array([0.2, 0.9]) + k) for k in range(2)]
        rs = [csdl.sum(csdl.sin(_Scale(2.0).evaluate(x))) for x in xs]
        ops = None
        if do_compress:
            first = compress(xs[0], rs[0])
            ops = [first, compress(xs[1], rs[1], share_with=first)]  # no assume_custom_ops_match needed
        f = _total(rs)
        g = csdl.derivative(f, xs)
        rec.stop()
        return ops, rec, [f.value] + [g[x].value for x in xs]

    _, _, expected = build(False)
    ops, rec, got = build(True)
    _assert_all_close(got, expected)
    assert ops[1].shared_from is ops[0]
    vjps = _vjp_ops(rec)
    assert len(vjps) == 2 and sum(op.shared_from is None for op in vjps) == 1


def test_share_with_mismatch_raises_before_changing_graph():
    rec = csdl.Recorder(inline=True)
    rec.start()
    x1 = csdl.Variable(value=np.array([0.3, 0.6]))
    x2 = csdl.Variable(value=np.array([0.3, 0.6]))
    r1, r2 = csdl.sum(csdl.sin(x1) ** 2.0), csdl.sum(csdl.sin(x2) ** 3.0)
    first = compress(x1, r1)
    before = op_names(rec)
    with pytest.raises(ValueError, match='does not match the region of share_with'):
        compress(x2, r2, share_with=first)
    with pytest.raises(TypeError, match='share_with must be'):
        compress(x2, r2, share_with='first')
    assert op_names(rec) == before
    rec.stop()


def test_share_with_takes_jit_kwargs_from_the_shared_operation():
    rec, c, xs, ts, rs = _blocks(2)
    first = compress([xs[0], ts[0]], rs[0], jit_kwargs={'keep_unused': True})
    with pytest.raises(ValueError, match='jit_kwargs differ'):
        compress([xs[1], ts[1]], rs[1], share_with=first, jit_kwargs={'keep_unused': False})
    second = compress([xs[1], ts[1]], rs[1], share_with=first)
    rec.stop()
    assert second.jit_kwargs == {'keep_unused': True} and second.shared_from is first


@pytest.mark.parametrize('how', ['automatic', 'share_with'])
def test_derivatives_share_when_blocks_are_compressed_as_they_are_recorded(how):
    # Compressing a block frees node indices that later blocks reuse, so later
    # blocks are numbered differently from the first. Sharing renumbers each
    # matched subgraph like its template, so CSDL records their derivative
    # graphs identically and those share one compiled function too.
    def build(do_compress):
        rec = csdl.Recorder(inline=True)
        rec.start()
        x = csdl.Variable(value=np.array([0.2, 0.5, 0.9]))
        h, first = x, None
        for _ in range(4):
            start = h
            for _ in range(5):
                h = csdl.tanh(0.9 * h + 0.1) + 0.05 * csdl.sin(h)
            if do_compress:
                op = compress(start, h, absorb_feeders=True, share_with=first if how == 'share_with' else None)
                first = first or op
        f = csdl.sum(h**2)
        g = csdl.derivative(f, x)
        rec.stop()
        return rec, [f.value, g.value]

    _, expected = build(False)
    rec, got = build(True)
    _assert_all_close(got, expected)
    vjps = _vjp_ops(rec)
    owners = {id(op.shared_from or op) for op in vjps}
    assert len(vjps) == 4 and len(owners) == 1
    rec.execute()
    owner = vjps[0].shared_from or vjps[0]
    assert owner.jit_fn._cache_size() == 1


def test_derivatives_of_shared_operations_share_without_matching(monkeypatch):
    import csdl_alpha.src.operations.compress_operations as compress_module
    rec, c, xs, ts, rs = _blocks(3)
    first = compress([xs[0], ts[0]], rs[0], share_compiled=False)
    for k in (1, 2):
        compress([ts[k], xs[k]], rs[k], share_with=first)

    calls = {'match': 0, 'trace': 0}
    real_match, real_trace = compress_module._match, compress_module._trace_region
    def counting_match(*args, **kwargs):
        calls['match'] += 1
        return real_match(*args, **kwargs)
    def counting_trace(*args, **kwargs):
        calls['trace'] += 1
        return real_trace(*args, **kwargs)
    monkeypatch.setattr(compress_module, '_match', counting_match)
    monkeypatch.setattr(compress_module, '_trace_region', counting_trace)

    f = _total(rs)
    csdl.derivative(f, xs + ts)
    csdl.derivative(f, xs + ts)  # a second derivative call reuses the first's derivative function
    rec.stop()
    assert calls == {'match': 0, 'trace': 0}
    vjps = _vjp_ops(rec)
    assert len(vjps) == 6 and len({id(op.shared_from or op) for op in vjps}) == 1
