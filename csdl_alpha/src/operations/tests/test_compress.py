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
