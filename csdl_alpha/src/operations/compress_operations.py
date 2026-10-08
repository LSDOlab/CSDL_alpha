from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
import csdl_alpha.utils.testing_utils as csdl_tests
import numpy as np

class CompressedOperation(SubgraphOperation):
    def __init__(self, subgraph, inputs, outputs, name, jax_jit=True):
        super().__init__(*inputs)
        self.name = name
        self.jax_jit = jax_jit
        self.assign_subgraph(subgraph)
        self.set_outputs(outputs)

        self.jax_function = None
    # def finalize_and_return_outputs(self):
    #     for output in self.outputs:
    #         self.recorder._add_node(output)

    #     outputs = super().finalize_and_return_outputs(skip_inline = True)
    #     if self.num_outputs == 1:
    #         return outputs
    #     return outputs

    def compute_jax(self, *args):
        from csdl_alpha.backends.jax import create_jax_function
        import jax

        if self.jax_function is None:
            # print('build')
            self.jax_function = create_jax_function(self.get_subgraph(), self.outputs, self.inputs)
            if self.jax_jit:
                self.jax_function = jax.jit(self.jax_function)
        else:
            pass
            # print('save')

        outs = tuple(self.jax_function(*args))
        return outs
    
    def compute_inline(self, *args):
        for i, input in enumerate(self.inputs):
            input.value = args[i]
        self.get_subgraph().execute_inline()
        if self.num_outputs == 1:
            return self.outputs[0].value
        else:
            return tuple([output.value for output in self.outputs])

    def evaluate_vjp(self, cotangents, *inputs_outputs):
        import csdl_alpha as csdl
        from csdl_alpha.src.operations.derivatives.reverse import vjp
        
        inputs = inputs_outputs[:len(self.inputs)]
        outputs = inputs_outputs[len(self.inputs):]

        seeds= []
        wrts = []
        for output in outputs:
            if cotangents.check(output):
                # print('output', output, cotangents[output])

                if cotangents[output] is not None:
                    seeds.append((output, cotangents[output]))
        for input in inputs:
            if cotangents.check(input):
                wrts.append(input)

        in_vjps = vjp(seeds, wrts, self.get_subgraph())

        for input, input_vjp in in_vjps.items():
            if input_vjp is not None:
                cotangents.accumulate(input, input_vjp)

            if cotangents[input] is None:
                cotangents.accumulate(input, csdl.Variable(value = np.zeros(input.shape)))

def compress_current_operations():
    import csdl_alpha as csdl
    recorder = csdl.get_current_recorder()
    current_graph = recorder.active_graph
    
    sources = []
    targets = []
    for node in current_graph.node_table:
        if isinstance(node, Variable):
            if current_graph.in_degree(node) == 0 and current_graph.out_degree(node) == 0:
                # pass
                sources.append(node)
                targets.append(node)
            elif current_graph.in_degree(node) == 0:
                sources.append(node)
            elif current_graph.out_degree(node) == 0:
                targets.append(node)
            else:
                pass
    
    S, S_inputs, S_outputs = current_graph.extract_subgraph(
        sources = sources,
        targets = targets,
        keep_variables=True,
    )
    compressed_operation = CompressedOperation(
        S,
        list(S_inputs),
        list(S_outputs),
        name='compressed_operation',
    )
    compressed_operation.finalize_and_return_outputs(skip_inline=True)
    return compressed_operation

def _resolve_device(device):
    """``None``, a ``jax.Device``, or a platform name (``'cpu'``, ``'gpu'``) -> ``jax.Device`` or ``None``."""
    if device is None or not isinstance(device, str):
        return device
    import jax
    try:
        return jax.devices(device)[0]
    except RuntimeError as error:
        raise ValueError(f"No JAX device for {device!r}: {error}") from None


class JaxCompressedOperation(CompressedOperation):
    """A ``CompressedOperation`` that runs as one jitted JAX function, derivatives included.

    The subgraph is translated with ``create_jax_function`` and wrapped in
    ``jax.jit(..., **jit_kwargs)`` once, on first use. Compared with
    ``CompressedOperation``:

    - Inline evaluation (``recorder.inline``, ``PySimulator``) calls that
      jitted function on ``device`` instead of executing the subgraph
      operation by operation.
    - Inside an enclosing JAX trace (``JaxSimulator``, an enclosing compressed
      operation), ``compute_jax`` either calls the function through a host
      callback, so that it stays its own executable on its own ``device``
      (``compile_separately=True``, the default), or traces it into the
      enclosing program, which XLA then compiles and optimizes as a whole.
    - Reverse mode records CSDL's own VJP of the subgraph into a copy of it
      and wraps that copy as another ``JaxCompressedOperation``. Every
      operation inside uses its CSDL derivative (custom operations call
      ``compute_derivatives``; implicit operations use their nonlinear
      solver's implicit derivative), the derivative is one compiled node on
      the graph, and it is differentiable in turn.

    Parameters
    ----------
    subgraph : Graph
        The graph to run. Its leaf variables are ``inputs`` or literal
        constants, which are compiled in.
    inputs, outputs : list[Variable]
        Variables of ``subgraph`` that become this operation's inputs and
        outputs. Repeated inputs are merged.
    name : str
        Operation name.
    device : str or jax.Device, optional
        Where the operation runs (``'cpu'``, ``'gpu'``, or a ``jax.Device``);
        by default JAX's default device. Inside an enclosing JAX trace this
        applies only with ``compile_separately``.
    jit_kwargs : dict, optional
        Extra keyword arguments for ``jax.jit``.
    compile_separately : bool
        Keep this operation out of any enclosing JAX program, by default True.
        The enclosing program then compiles without it, which cuts its compile
        time and memory, and this operation runs on ``device``. Each call costs
        a round trip through host memory and loses XLA optimization across the
        boundary; under ``vmap`` the callback runs once per batch element. With
        False, the operation is traced into the enclosing program instead.

    Derivative operations inherit ``device``, ``jit_kwargs``, and
    ``compile_separately``.
    """

    def __init__(
            self,
            subgraph,
            inputs,
            outputs,
            name = 'jax_compressed',
            device = None,
            jit_kwargs = None,
            compile_separately = True,
        ):
        import jax
        jax.config.update("jax_enable_x64", True)
        super().__init__(subgraph, list(dict.fromkeys(inputs)), list(outputs), name)
        self.device = _resolve_device(device)
        self.jit_kwargs = dict(jit_kwargs or {})
        self.compile_separately = compile_separately

    @property
    def jit_fn(self):
        """The subgraph as a ``jax.jit`` function, built on first use."""
        if self.jax_function is None:
            import jax
            from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
            fn = create_jax_function(self.get_subgraph(), self.outputs, self.inputs)
            self.jax_function = jax.jit(fn, **self.jit_kwargs)
        return self.jax_function

    def compute_jax(self, *args):
        if not self.compile_separately:
            return tuple(self.jit_fn(*args))
        import jax
        from csdl_alpha.backends.jax.utils import host_callback
        result_shapes = [jax.ShapeDtypeStruct(var.shape, np.float64) for var in self.outputs]
        return tuple(host_callback(self._compute_on_host, result_shapes, *args))

    def _compute_on_host(self, *args):
        outs = self.compute_inline(*args)
        return (outs,) if self.num_outputs == 1 else outs

    def compute_inline(self, *args):
        if self.device is not None:
            import jax
            args = jax.device_put(args, self.device)
        outs = self.jit_fn(*args)
        outs = tuple(np.array(out).reshape(var.shape) for out, var in zip(outs, self.outputs))
        if self.num_outputs == 1:
            return outs[0]
        return outs

    def evaluate_vjp(self, cotangents, *inputs_outputs):
        import csdl_alpha as csdl
        from csdl_alpha.src.operations.derivatives.reverse import vjp

        inputs = inputs_outputs[:self.num_inputs]
        outputs = inputs_outputs[self.num_inputs:]
        # Outputs with a None cotangent contribute nothing; leave them out.
        seeds = [(y, cotangents[y]) for y in outputs if cotangents.check(y) and cotangents[y] is not None]
        wrts = [x for x in inputs if cotangents.check(x)]
        if not seeds or not wrts:
            return

        # Record CSDL's reverse mode into a copy of the subgraph, so the subgraph
        # itself stays a clean forward graph for later derivative calls. The
        # recorder creates the copy so that it sits in its graph tree, which
        # operations that open their own subgraphs (loops, composed ops) need.
        # The outer cotangent variables join the copy as its seed inputs.
        recorder = csdl.get_current_recorder()
        inline = recorder.inline
        recorder.inline = False
        recorder._enter_subgraph(name=f'vjp_{self.name}')
        work = recorder.active_graph
        work.rxgraph = self.get_subgraph().rxgraph.copy()
        work.update_node_table()
        try:
            for _, seed in seeds:
                work.add_node(seed)
            input_cotangents = vjp(seeds, wrts, work)
            # Fresh variables, so each gradient is produced inside the copy.
            grads = [csdl.copyvar(input_cotangents[x]) if input_cotangents[x] is not None
                     else csdl.copyvar(csdl.Variable(value=np.zeros(x.shape))) for x in wrts]
        finally:
            recorder._exit_subgraph()
            recorder.inline = inline

        for grad in grads:
            recorder._add_node(grad)
        # The copy also holds forward operations the VJP does not need; jax.jit drops them.
        vjp_op = JaxCompressedOperation(
            work, list(inputs) + [seed for _, seed in seeds], grads, name=f'vjp_{self.name}',
            device=self.device, jit_kwargs=self.jit_kwargs, compile_separately=self.compile_separately)
        vjp_op.finalize_and_return_outputs()
        for x, grad in zip(wrts, grads):
            cotangents.accumulate(x, grad)


def compress(
        inputs,
        outputs,
        name = 'jax_compressed',
        absorb_feeders = False,
        intermediates = 'none',
        device = None,
        jit_kwargs = None,
        compile_separately = True,
    ):
    """Replace the operations between ``inputs`` and ``outputs`` with one JAX-compiled operation.

    The compressed region is every operation that both depends on ``inputs``
    and feeds ``outputs`` (plus, with ``absorb_feeders``, upstream operations
    that only feed it). Those operations and their intermediate variables move
    out of the active graph into the subgraph of a single
    :class:`JaxCompressedOperation`. That operation runs the region as one
    ``jax.jit`` function and differentiates it with CSDL's own reverse mode,
    compiled the same way. The ``outputs`` variables are kept: the new operation
    produces them, so downstream operations, constraints, objectives, and
    simulator handles still work.

    Any other variable the region reads (a parameter, or a value computed
    upstream) becomes an extra input whose value is read from the graph, so it
    is never frozen and is differentiable. Only literal constants
    (``csdl.Constant``, such as the ``2.0`` in ``2.0 * x``) are compiled in.

    Parameters
    ----------
    inputs : Variable or list[Variable]
        Where the region starts. No input may depend on another.
    outputs : Variable or list[Variable]
        Where the region ends.
    name : str
        Name of the new operation.
    absorb_feeders : bool
        Also compress upstream operations whose outputs only the region uses,
        such as the broadcast of a scalar constant, by default False. Each one
        left outside stays a separate operation.
    intermediates : str
        Which intermediates also become outputs. ``'none'`` (default): only
        ``outputs``; an intermediate used outside the region (or registered as
        a constraint or objective) raises. ``'used'``: add those
        intermediates as outputs. ``'all'``: every variable the region
        computes is an output.
    device : str or jax.Device, optional
        Where the operation runs (``'cpu'``, ``'gpu'``, or a ``jax.Device``).
        Under ``JaxSimulator`` this applies only with ``compile_separately``.
    jit_kwargs : dict, optional
        Extra keyword arguments for ``jax.jit``.
    compile_separately : bool
        Under ``JaxSimulator``, call this operation's own compiled function
        through a host callback instead of compiling it into the simulator's
        program, by default True. Set False to let XLA compile and optimize
        the region together with the rest of the program. See
        :class:`JaxCompressedOperation`.

    Returns
    -------
    JaxCompressedOperation
        ``op.inputs`` is ``inputs`` followed by ``op.extra_inputs``,
        ``op.outputs`` is ``outputs`` followed by ``op.extra_outputs``
        (intermediates added by ``intermediates``), and ``op.get_subgraph()``
        is the region.

    Examples
    --------
    >>> recorder = csdl.Recorder(inline=True)
    >>> recorder.start()
    >>> x = csdl.Variable(value=np.array([0.1, 0.2]))
    >>> z = csdl.sum(csdl.sin(x) * x)
    >>> op = csdl.experimental.compress(x, z)
    >>> f = 2.0 * z
    >>> csdl.derivative(f, x).value
    array([[0.39866767, 0.78936529]])
    """
    import csdl_alpha as csdl
    from csdl_alpha.src.graph.variable import Constant
    from csdl_alpha.utils.inputs import listify_variables

    if intermediates not in ('none', 'used', 'all'):
        raise ValueError(f"intermediates must be 'none', 'used', or 'all', not {intermediates!r}.")
    device = _resolve_device(device)
    inputs = listify_variables(inputs)
    outputs = listify_variables(outputs)
    for variables, label in [(inputs, 'inputs'), (outputs, 'outputs')]:
        if len(set(variables)) != len(variables):
            raise ValueError(f"`{label}` contains a variable more than once.")

    recorder = csdl.get_current_recorder()
    graph = recorder.active_graph
    for var in inputs + outputs:
        if var not in graph.node_table:
            raise ValueError(f"{var.info()} is not in the active graph.")
    for var in outputs:
        if var in inputs:
            raise ValueError(f"{var.info()} is given as both an input and an output.")

    ops, internal, extra_outputs = _find_region(graph, inputs, outputs, absorb_feeders, intermediates)
    outputs = outputs + extra_outputs

    # Gather the variables the region reads that it does not compute. Literal
    # constants cannot change, so they are compiled into the function;
    # everything else becomes an input.
    input_set = set(inputs)
    extra_inputs = []
    constants = []
    seen = set()
    for op in ops:
        for var in op.inputs:
            if var not in internal and var not in input_set and var not in seen:
                seen.add(var)
                is_literal = type(var) is Constant and var.value is not None
                (constants if is_literal else extra_inputs).append(var)
    op_inputs = inputs + extra_inputs

    # Move the region's operations into their own graph (this deletes them from
    # `graph` and keeps the variables there), then drop the intermediates and
    # any constants nothing else uses. The outputs stay; the new operation
    # produces them.
    region = graph.extract_subgraph_nodes(set(ops) | internal | set(op_inputs) | set(constants))
    region.name = name
    output_set = set(outputs)
    graph._delete_nodes(
        [var for var in internal if var not in output_set]
        + [var for var in constants if graph.out_degree(var) == 0]
    )

    compressed = JaxCompressedOperation(
        region, op_inputs, outputs, name=name, device=device,
        jit_kwargs=jit_kwargs, compile_separately=compile_separately)
    compressed.extra_inputs = extra_inputs
    compressed.extra_outputs = extra_outputs
    # Skip inline evaluation: the outputs already hold the values the region computed.
    compressed.finalize_and_return_outputs(skip_inline=True)
    return compressed


def _find_region(graph, inputs, outputs, absorb_feeders, intermediates):
    """Return the region's operations in topological order, the variables they
    compute, and the intermediates to add as outputs."""
    import csdl_alpha as csdl
    import rustworkx as rx
    from csdl_alpha.src.operations.operation_subclasses import RandomOperation

    rxgraph = graph.rxgraph
    input_indices = {graph.node_table[var] for var in inputs}
    for var in inputs:
        if input_indices & rx.ancestors(rxgraph, graph.node_table[var]):
            raise ValueError(f"Input {var.info()} depends on another input.")

    # Descendants of the inputs that are ancestors of the outputs. This raises
    # if an input affects no output or an output depends on no input.
    between = graph._get_intersection(
        inputs, outputs, add_hanging_input_variables=False, add_hanging_output_variables=False)
    op_indices = {i for i in between if isinstance(rxgraph[i], Operation)}

    recorder = csdl.get_current_recorder()
    registered = set(recorder.constraints) | set(recorder.objectives)
    if absorb_feeders:
        _absorb_feeders(graph, op_indices, input_indices, registered)

    order = rx.topological_sort(rxgraph)
    ops = [rxgraph[i] for i in order if i in op_indices]
    internal = {var for op in ops for var in op.outputs}

    for op in ops:
        if isinstance(op, RandomOperation):
            raise NotImplementedError(f"Cannot compress random operation {op.info()}.")

    # An intermediate used outside the region would lose its producer, so it
    # must become an output.
    output_set = set(outputs)
    extra_outputs = []
    for var in (var for op in ops for var in op.outputs):
        if var in output_set or var in extra_outputs:
            continue
        if intermediates == 'all':
            extra_outputs.append(var)
            continue
        outside = [rxgraph[i] for i in rxgraph.successor_indices(graph.node_table[var]) if i not in op_indices]
        if outside or var in registered:
            if intermediates == 'used':
                extra_outputs.append(var)
                continue
            users = ', '.join(op.info() for op in outside) or 'a constraint or objective'
            raise ValueError(
                f"Intermediate {var.info()} is used outside the compressed region (by {users}). "
                "Add it to `outputs`, or pass intermediates='used'."
            )
    return ops, internal, extra_outputs


def _absorb_feeders(graph, op_indices, input_indices, registered):
    """Add to ``op_indices`` every upstream operation whose outputs are used only inside the region."""
    rxgraph = graph.rxgraph

    def producers_of_inputs(op_index):
        for var_index in rxgraph.predecessor_indices(op_index):
            if var_index not in input_indices:
                yield from rxgraph.predecessor_indices(var_index)

    stack = [p for i in op_indices for p in producers_of_inputs(i)]
    while stack:
        candidate = stack.pop()
        if candidate in op_indices:
            continue
        exclusive = all(
            rxgraph[out] not in registered
            and all(consumer in op_indices for consumer in rxgraph.successor_indices(out))
            for out in rxgraph.successor_indices(candidate)
        )
        if exclusive:
            op_indices.add(candidate)
            stack.extend(producers_of_inputs(candidate))


class TestCompressOp(csdl_tests.CSDLTest):
    @staticmethod
    def simple_model(x, y):
        import csdl_alpha as csdl
        return x*y + 3*x*y**2 + 5*x**2*y + 7*x**2*y**2
    
    @staticmethod
    def d_simple_model_dx(x, y):
        return y + 3*y**2 + 10*x*y + 14*x*y**2

    def test_simple(self):
        import csdl_alpha as csdl

        self.prep()
        x = csdl.Variable(name='x', value=2.0)
        y = csdl.Variable(name='y', value=3.0)
        z = self.simple_model(x, y)
        z_np = self.simple_model(x.value, y.value)

        op = compress_current_operations()
        assert z in op.outputs
        compare_values = []
        compare_values += [csdl_tests.TestingPair(z, z_np, tag = 'simple')]
        self.run_tests(compare_values, verify_derivatives=True)


class TestJaxCompress(csdl_tests.CSDLTest):
    """compress() under every backend and derivative mode of the test matrix."""

    x_val = np.array([0.1, 0.5, -0.3, 1.2])
    y_val = np.array([1.0, 2.0, 0.5, -1.0])
    c_val = np.array([0.7, -0.2])

    @staticmethod
    def region(x, y, c, lib):
        """Elementwise ops, a matvec and a reduction; `lib` is csdl or a numpy shim."""
        a = lib.sin(x) * y + lib.exp(0.3 * x)
        b = lib.matvec(lib.reshape(a, (2, 2)), c)
        return a, lib.sum(b**2) + lib.norm(a)

    class np_lib:
        sin, exp, sum = np.sin, np.exp, np.sum
        reshape = staticmethod(np.reshape)
        matvec = staticmethod(lambda A, v: A @ v)
        norm = staticmethod(lambda a: np.array([np.linalg.norm(a)]))

    def build(self, **compress_kwargs):
        import csdl_alpha as csdl
        x = csdl.Variable(name='x', value=self.x_val)
        y = csdl.Variable(name='y', value=self.y_val)
        c = csdl.Variable(name='c', value=self.c_val)  # read by the region but not passed to compress
        a, z = self.region(x, y, c, csdl)
        return x, y, c, a, z

    def expected(self):
        a, z = self.region(self.x_val, self.y_val, self.c_val, self.np_lib)
        return a, np.array([2.0 * z.item() + np.sum(self.x_val)])

    def test_docstring(self):
        self.docstest(compress)

    def test_options(self):
        option_sets = [
            {},
            {'absorb_feeders': True},
            {'compile_separately': False},
            {'absorb_feeders': True, 'compile_separately': False, 'jit_kwargs': {'keep_unused': True}},
        ]
        import csdl_alpha as csdl
        for options in option_sets:
            self.prep()
            x, y, c, a, z = self.build()
            op = compress([x, y], z, **options)
            assert isinstance(op, JaxCompressedOperation) and op.outputs == [z]
            f = 2.0 * z + csdl.sum(x)
            _, f_np = self.expected()
            self.run_tests([csdl_tests.TestingPair(f, f_np, tag=str(options))], verify_derivatives=True)

    def test_intermediates_used_outside(self):
        import csdl_alpha as csdl
        self.prep()
        x, y, c, a, z = self.build()
        w = csdl.sum(a * 3.0)  # uses the intermediate `a` outside x, y -> z
        op = compress([x, y], z, intermediates='used')
        assert op.extra_outputs == [a]
        a_np, f_np = self.expected()
        self.run_tests([
            csdl_tests.TestingPair(2.0 * z + csdl.sum(x), f_np, tag='f'),
            csdl_tests.TestingPair(w, np.array([3.0 * np.sum(a_np)]), tag='w'),
        ], verify_derivatives=True)

    def test_implicit_operation(self):
        import csdl_alpha as csdl
        a_val = np.array([2.0, 3.0])
        for compile_separately in [False, True]:
            self.prep()
            a = csdl.Variable(name='a', value=a_val)
            x = csdl.ImplicitVariable(name='x', value=np.ones(2))
            solver = csdl.nonlinear_solvers.Newton(print_status=False, tolerance=1e-12)
            solver.add_state(x, x**3 + a * x - 10.0)
            solver.run()
            z = csdl.sum(x**2 * a)
            compress(a, z, compile_separately=compile_separately)
            x_np = np.array([np.real(r[np.isreal(r)][0]) for r in (np.roots([1, 0, ai, -10]) for ai in a_val)])
            self.run_tests(
                [csdl_tests.TestingPair(z, np.array([np.sum(x_np**2 * a_val)]), decimal=9)],
                verify_derivatives=True)

    def test_custom_operation(self):
        import csdl_alpha as csdl

        class CubePlusProduct(csdl.CustomExplicitOperation):
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

        x_val, y_val = np.array([0.3, -0.4, 0.9]), np.array([1.0, 0.5, -2.0])
        self.prep()
        x = csdl.Variable(name='x', value=x_val)
        y = csdl.Variable(name='y', value=y_val)
        u = csdl.sin(x) * 2.0
        z = csdl.sum(csdl.exp(0.1 * CubePlusProduct().evaluate(u, y)) * u)
        compress([x, y], z)
        u_np = np.sin(x_val) * 2.0
        z_np = np.sum(np.exp(0.1 * (u_np**3 + u_np * y_val)) * u_np)
        self.run_tests([csdl_tests.TestingPair(z, np.array([z_np]))], verify_derivatives=True)

    def test_loop_and_nested(self):
        import csdl_alpha as csdl
        x_val = np.array([0.2, 0.4, 0.8])
        self.prep()
        x = csdl.Variable(name='x', value=x_val)
        acc = csdl.Variable(value=np.zeros(3))
        for i in csdl.frange(3):
            acc = acc + csdl.sin(x * (i + 1.0))
        z = csdl.sum(acc * x)
        compress(x, z)                                         # inner, separate
        f = csdl.exp(z * 0.1)
        compress(x, f, name='outer', compile_separately=False)  # fused, contains the inner
        z_np = np.sum(sum(np.sin(x_val * (i + 1.0)) for i in range(3)) * x_val)
        self.run_tests([csdl_tests.TestingPair(f, np.array([np.exp(0.1 * z_np)]))], verify_derivatives=True)

    def test_compress_inside_loop_body(self):
        import csdl_alpha as csdl
        x_val, p_val = np.array([0.2, 0.5, 0.9]), np.array([1.3])

        def step(h, i, p, lib):
            return lib.tanh(lib.sin(h * p) + (i + 1.0) * 0.3 * h) * 0.9 + h * 0.1

        h_np = x_val
        for i in range(4):
            h_np = step(h_np, i, p_val, np)
        f_np = np.array([np.sum(h_np**2)])

        for loop_kind in ['frange', 'enter_loop']:
            for compile_separately in [True, False]:
                self.prep()
                x = csdl.Variable(name='x', value=x_val)
                p = csdl.Variable(name='p', value=p_val)
                if loop_kind == 'frange':
                    h = x
                    for i in csdl.frange(4):
                        h_new = step(h, i, p, csdl)
                        compress(h, h_new, compile_separately=compile_separately)
                        h = h_new
                else:
                    with csdl.experimental.enter_loop(vals=[list(range(4))]) as loop_builder:
                        i = loop_builder.get_loop_indices()
                        h0 = loop_builder.initialize_feedback(x)
                        h1 = step(h0, i, p, csdl)
                        compress(h0, h1, compile_separately=compile_separately)
                        loop_builder.finalize_feedback(h0, h1)
                    h = loop_builder.add_output(h1)
                    loop_builder.finalize()
                f = csdl.sum(h**2)
                self.run_tests(
                    [csdl_tests.TestingPair(f, f_np, tag=f'{loop_kind} separate={compile_separately}')],
                    verify_derivatives=True)


if __name__ == '__main__':
    test = TestCompressOp()
    test.overwrite_backend = 'jax'
    test.test_simple()