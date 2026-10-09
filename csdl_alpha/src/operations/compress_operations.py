from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
import csdl_alpha.utils.testing_utils as csdl_tests
import numpy as np
import re

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

    share_compiled : bool
        Reuse the compiled function of an earlier ``JaxCompressedOperation`` in
        this recorder whose subgraph is identical (same operations, parameters,
        shapes, wiring, and compiled-in constants), by default True. The
        earlier operation is ``shared_from``.
    assume_custom_ops_match : bool
        Treat operations that JAX runs through a Python callback (custom
        operations, operations without a JAX implementation) as identical when
        their class, shapes, and JAX trace match, by default False. JAX cannot
        see the Python state such an operation computes with (for example a
        parameter stored on the instance), so by default subgraphs containing
        one are never shared. Set True only when you know those operations
        compute the same function.

    Derivative operations inherit ``device``, ``jit_kwargs``,
    ``compile_separately``, ``share_compiled``, and
    ``assume_custom_ops_match``.
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
            share_compiled = True,
            assume_custom_ops_match = False,
        ):
        import jax
        jax.config.update("jax_enable_x64", True)
        super().__init__(subgraph, list(dict.fromkeys(inputs)), list(outputs), name)
        self.device = _resolve_device(device)
        self.jit_kwargs = dict(jit_kwargs or {})
        self.compile_separately = compile_separately
        self.share_compiled = share_compiled
        self.assume_custom_ops_match = assume_custom_ops_match
        # The operation whose compiled function this one reuses, and, for each
        # of that operation's inputs, the position of the matching input here.
        self.shared_from = None
        self._input_permutation = None
        self._share_group = _ShareGroup(self)
        if share_compiled:
            _share_with_identical(self)

    @property
    def jit_fn(self):
        """The subgraph as a ``jax.jit`` function, built on first use."""
        if self.jax_function is None:
            import jax
            if self.shared_from is not None:
                shared_fn = self.shared_from.jit_fn
                permutation = self._input_permutation
                if permutation == list(range(len(permutation))):
                    self.jax_function = shared_fn
                else:
                    self.jax_function = lambda *args: shared_fn(*(args[j] for j in permutation))
            else:
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
        if self._input_permutation is not None:
            # List inputs in the order of the operation that owns the compiled
            # function, so derivative operations of operations that share it
            # come out in one order and can share too.
            inputs = [inputs[j] for j in self._input_permutation]
        # Outputs with a None cotangent contribute nothing; leave them out.
        seeded = [j for j, y in enumerate(outputs) if cotangents.check(y) and cotangents[y] is not None]
        differentiated = [i for i, x in enumerate(inputs) if cotangents.check(x)]
        if not seeded or not differentiated:
            return
        seeds = [(outputs[j], cotangents[outputs[j]]) for j in seeded]
        wrts = [inputs[i] for i in differentiated]

        # Every operation in this group computes the same function, so a
        # derivative operation another member made for the same seeded outputs
        # and differentiated inputs computes the same function too: share its
        # compiled function without checking. (Repeated variables among the
        # inputs and seeds would change the derivative operation's inputs.)
        vjp_inputs = list(inputs) + [seed for _, seed in seeds]
        pattern = (tuple(seeded), tuple(differentiated)) if len(set(vjp_inputs)) == len(vjp_inputs) else None
        shared_vjp = self._share_group.derivatives.get(pattern) if pattern is not None else None

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
            work, vjp_inputs, grads, name=f'vjp_{self.name}',
            device=self.device, jit_kwargs=self.jit_kwargs, compile_separately=self.compile_separately,
            share_compiled=False, assume_custom_ops_match=self.assume_custom_ops_match)
        vjp_op.share_compiled = self.share_compiled
        if shared_vjp is not None:
            _share_explicitly(vjp_op, shared_vjp, list(range(len(vjp_inputs))))
        elif pattern is not None:
            self._share_group.derivatives[pattern] = vjp_op
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
        share_compiled = True,
        find_repeats = False,
        assume_custom_ops_match = False,
        share_with = None,
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
    share_compiled : bool
        Reuse the compiled function of an earlier compressed operation in this
        recorder whose region is identical, by default True. Its derivative
        operations share in the same way.
    find_repeats : bool
        Also find every other copy of this region in the active graph (same
        operations, parameters, shapes, wiring, and constants, reading any
        inputs) and compress each one with the same compiled function, by
        default False. Copies that have an intermediate used outside them or
        would create a cycle are skipped. The copies are ``op.repeats``.

        Copies can overlap (in a chain ``x1 = sin(x0)``, ``x2 = sin(x1)``, ...,
        a two-``sin`` region matches at every offset). The region given here
        is always compressed as given, and copies are then taken greedily in
        topological order of their outputs, skipping any that would reuse an
        operation already compressed. This gives the most copies when
        overlapping copies line up like a chain, but not necessarily in
        branching graphs. To choose the copies exactly, compress each one and
        pass ``share_with``.
    assume_custom_ops_match : bool
        Let regions containing custom operations (or other operations JAX runs
        through a Python callback) count as identical when the operations'
        class, shapes, and JAX trace match, by default False. JAX cannot see
        the Python state a custom operation computes with, so only set this
        when you know the custom operations in matching regions compute the
        same function.
    share_with : JaxCompressedOperation, optional
        An earlier compressed operation whose compiled function this region
        should reuse. The region is checked against it (same operations,
        parameters, shapes, wiring, and constants; ``outputs`` in the same
        order; inputs in any order) and a mismatch raises before the graph is
        changed. Custom operations of the same class count as matching, since
        naming ``share_with`` asserts that the regions compute the same
        function. Derivative operations share as well. ``jit_kwargs`` are
        taken from ``share_with``.

    Returns
    -------
    JaxCompressedOperation
        ``op.inputs`` is ``inputs`` followed by ``op.extra_inputs``,
        ``op.outputs`` is ``outputs`` followed by ``op.extra_outputs``
        (intermediates added by ``intermediates``), ``op.get_subgraph()``
        is the region, ``op.shared_from`` is the operation whose compiled
        function it reuses (or None), and ``op.repeats`` lists the copies
        compressed by ``find_repeats``.

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
    from csdl_alpha.utils.inputs import listify_variables

    if intermediates not in ('none', 'used', 'all'):
        raise ValueError(f"intermediates must be 'none', 'used', or 'all', not {intermediates!r}.")
    if share_with is not None:
        if not isinstance(share_with, JaxCompressedOperation):
            raise TypeError(f"share_with must be a JaxCompressedOperation, not {type(share_with).__name__}.")
        if jit_kwargs is not None and dict(jit_kwargs) != share_with.jit_kwargs:
            raise ValueError("jit_kwargs differ from share_with's; the shared compiled function uses share_with's.")
        jit_kwargs = share_with.jit_kwargs
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
                (constants if _is_literal(var) else extra_inputs).append(var)
    op_inputs = inputs + extra_inputs

    if share_with is not None:
        share_permutation = _match(share_with, graph, op_inputs, outputs, set(ops), assume_custom_ops_match=True)
        if share_permutation is None:
            raise ValueError(
                f"The region does not match the region of share_with ({share_with.info()}): they differ in "
                "operations, parameters, shapes, wiring, constants, or the order of `outputs`.")

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
        jit_kwargs=jit_kwargs, compile_separately=compile_separately,
        share_compiled=share_compiled and share_with is None, assume_custom_ops_match=assume_custom_ops_match)
    if share_with is not None:
        compressed.share_compiled = share_compiled  # for its derivative operations
        _share_explicitly(compressed, share_with, share_permutation)
    compressed.extra_inputs = extra_inputs
    compressed.extra_outputs = extra_outputs
    # Skip inline evaluation: the outputs already hold the values the region computed.
    compressed.finalize_and_return_outputs(skip_inline=True)
    compressed.repeats = _compress_repeats(compressed, graph, len(inputs)) if find_repeats else []
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


# ---- Sharing compiled functions between identical regions ----------------------
#
# Two regions are identical when a one-to-one map pairs their operations and
# variables such that paired operations are of the same class with the same
# shapes, every operation input is wired from the paired producer output at the
# same port, and compiled-in constants have equal values (the structural walk),
# and the two regions, traced with JAX operation by operation in paired order,
# give the same jaxpr (which checks parameters the walk cannot see, such as
# reshape shapes or exponents). Template inputs may map to any variables.
# Because operation inputs are ordered and each variable has one producer, the
# walk goes back from the outputs port by port, with no search except where an
# operation is reachable only forward from a variable (see _complete).
#
# Operations that share a compiled function form a _ShareGroup. Their
# derivative operations are shared within the group without matching, since
# they differentiate the same function.


class _ShareGroup:
    """Operations that compute one function, and the derivative operations made
    for them, keyed by (seeded outputs, differentiated inputs) in the order of
    the operation that owns the compiled function."""

    def __init__(self, owner):
        self.members = [owner]
        self.derivatives = {}


def _is_literal(var):
    from csdl_alpha.src.graph.variable import Constant
    return type(var) is Constant and var.value is not None


def _registry_key(op):
    num_ops = sum(isinstance(node, Operation) for node in op.get_subgraph().node_table)
    return (
        tuple(sorted(var.shape for var in op.inputs)),
        tuple(var.shape for var in op.outputs),
        num_ops,
        repr(sorted(op.jit_kwargs.items())),
    )


def _root(op):
    return op.shared_from if op.shared_from is not None else op


def _share_explicitly(op, template, permutation):
    """Make ``op`` use ``template``'s compiled function; ``permutation[i]`` is the
    position in ``op.inputs`` of ``template``'s input ``i``."""
    if template.shared_from is not None:
        # Compose with the template's own mapping onto the function's owner.
        permutation = [permutation[j] for j in template._input_permutation]
    op.shared_from = _root(template)
    op._input_permutation = permutation
    template._share_group.members.append(op)
    op._share_group = template._share_group


def _share_with_identical(op):
    """Point ``op`` at the compiled function of an identical earlier operation, or register it."""
    entries = op.recorder.compressed_operations.setdefault(_registry_key(op), [])
    for template in entries:
        permutation = _match(template, op.get_subgraph(), op.inputs, op.outputs, None, op.assume_custom_ops_match)
        if permutation is not None:
            _share_explicitly(op, template, permutation)
            return
    entries.append(op)


def _same_structure(t_op, c_op, ctx):
    """Whether two operations can pair in the walk (class and shapes); their
    parameters are compared later through the region trace."""
    if type(t_op) is not type(c_op):
        return False
    if len(t_op.inputs) != len(c_op.inputs) or len(t_op.outputs) != len(c_op.outputs):
        return False
    if any(a.shape != b.shape for a, b in zip(t_op.inputs + t_op.outputs, c_op.inputs + c_op.outputs)):
        return False
    if isinstance(t_op, JaxCompressedOperation):
        # Nested compressed operations sharing one compiled function, with the
        # same input order, are identical: trace them as a placeholder.
        if _root(t_op) is _root(c_op) and t_op._input_permutation == c_op._input_permutation:
            ctx.placeholders.add(t_op)
    return True


class _MatchState:
    def __init__(self):
        self.var_map, self.op_map, self.used_vars, self.used_ops = {}, {}, set(), set()

    def copy(self):
        new = _MatchState()
        new.var_map, new.op_map = dict(self.var_map), dict(self.op_map)
        new.used_vars, new.used_ops = set(self.used_vars), set(self.used_ops)
        return new


class _MatchContext:
    def __init__(self, template, c_graph, input_ok, leaf_ok, op_ok, assume_custom_ops_match):
        self.template = template
        self.t_graph = template.get_subgraph()
        self.t_inputs = set(template.inputs)
        self.t_ops = _live_ops(template)
        self.c_graph, self.input_ok, self.leaf_ok, self.op_ok = c_graph, input_ok, leaf_ok, op_ok
        self.assume = assume_custom_ops_match
        self.placeholders = set()  # template operations traced as placeholders


def _live_ops(op):
    """Operations in ``op``'s subgraph that its outputs depend on, in topological order."""
    if '_compress_live_ops' not in op.__dict__:
        import rustworkx as rx
        graph = op.get_subgraph()
        indices = set()
        for var in op.outputs:
            indices |= rx.ancestors(graph.rxgraph, graph.node_table[var])
        op._compress_live_ops = [graph.rxgraph[i] for i in rx.topological_sort(graph.rxgraph)
                                 if i in indices and isinstance(graph.rxgraph[i], Operation)]
    return op._compress_live_ops


def _walk(state, pairs, ctx):
    """Extend ``state`` by pairing template and candidate variables back to their producers; None on mismatch."""
    stack = list(pairs)
    while stack:
        t, c = stack.pop()
        if t in state.var_map:
            if state.var_map[t] is not c:
                return None
            continue
        if c in state.used_vars or t.shape != c.shape:
            return None
        state.var_map[t] = c
        state.used_vars.add(c)
        if t in ctx.t_inputs:
            if not ctx.input_ok(c):
                return None
            continue
        t_producers = ctx.t_graph.predecessors(t)
        if not t_producers:  # a value compiled into the template's function
            if not ctx.leaf_ok(c) or not np.array_equal(t.value, c.value):
                return None
            continue
        t_op = t_producers[0]
        c_producers = ctx.c_graph.predecessors(c) if c in ctx.c_graph.node_table else []
        if not c_producers:
            return None
        c_op = c_producers[0]
        if t_op in state.op_map:
            if state.op_map[t_op] is not c_op:
                return None
            continue
        if c_op in state.used_ops or not ctx.op_ok(c_op) or not _same_structure(t_op, c_op, ctx):
            return None
        state.op_map[t_op] = c_op
        state.used_ops.add(c_op)
        stack.extend(zip(t_op.outputs, c_op.outputs))
        stack.extend(zip(t_op.inputs, c_op.inputs))
    return state


def _complete(state, ctx):
    """Map template operations the backward walk did not reach, by trying the
    consumers of an already-mapped variable at the same input port."""
    missing = [op for op in ctx.t_ops if op not in state.op_map]
    if not missing:
        return state
    for t_op in missing:
        bound = [(port, var) for port, var in enumerate(t_op.inputs) if var in state.var_map]
        if bound:
            break
    else:
        return None
    port, t_var = bound[0]
    c_var = state.var_map[t_var]
    graph = ctx.c_graph
    for c_op in graph.rxgraph.successors(graph.node_table[c_var]):
        if len(c_op.inputs) > port and c_op.inputs[port] is c_var and type(c_op) is type(t_op):
            trial = _walk(state.copy(), list(zip(t_op.outputs, c_op.outputs)), ctx)
            if trial is not None and trial.op_map.get(t_op) is c_op:
                result = _complete(trial, ctx)
                if result is not None:
                    return result
    return None


def _jax_outputs(op, args):
    """``op``'s outputs as JAX arrays, as ``create_jax_function`` evaluates it."""
    from csdl_alpha.src.operations.loops.new_loop.new_loop import NewLoop
    if isinstance(op, JaxCompressedOperation):
        return list(op.jit_fn(*args))  # its own trace, not the callback of compile_separately
    if isinstance(op, NewLoop):
        outputs = {var: None for var in op.outputs}
        op.evaluate_jax(dict(zip(op.inputs, args)), outputs=outputs)
        return [outputs[var] for var in op.outputs]
    outs = op.compute_jax(*args)
    return list(outs) if isinstance(outs, (tuple, list)) else [outs]


def _trace_region(ops, inputs, outputs, placeholders):
    """``(jaxpr text, constants)`` of running ``ops`` in the given order, or None if it cannot be traced."""
    import jax
    import jax.numpy as jnp

    def region(*args):
        env = dict(zip(inputs, args))
        for op in ops:
            for var in op.inputs:
                if var not in env:  # a value compiled in
                    env[var] = jnp.asarray(var.value)
            if op in placeholders:
                outs = [jnp.zeros(var.shape) for var in op.outputs]
            else:
                outs = _jax_outputs(op, [env[var] for var in op.inputs])
            for var, out in zip(op.outputs, outs):
                env[var] = jnp.reshape(out, var.shape)
        return [env[var] for var in outputs]

    try:
        closed = jax.make_jaxpr(region)(*(jax.ShapeDtypeStruct(var.shape, np.float64) for var in inputs))
    except Exception:
        return None
    # Some JAX versions print a callback's Python function with its memory
    # address, which differs between otherwise identical regions. Whether a
    # callback may match at all is decided separately (assume_custom_ops_match).
    text = re.sub(r'0x[0-9a-fA-F]+', '0x?', str(closed.jaxpr))
    return text, tuple((np.shape(c), np.asarray(c).tobytes()) for c in closed.consts)


def _traces_match(state, ctx, c_inputs):
    """Whether the candidate paired in ``state`` traces like the template;
    ``c_inputs`` are the candidate inputs in the template's input order."""
    template = ctx.template
    cache = template.__dict__.setdefault('_compress_traces', {})
    key = frozenset(id(op) for op in ctx.placeholders)
    if key not in cache:
        cache[key] = _trace_region(ctx.t_ops, template.inputs, template.outputs, ctx.placeholders)
    t_trace = cache[key]
    c_ops = [state.op_map[op] for op in ctx.t_ops]
    c_placeholders = {state.op_map[op] for op in ctx.placeholders}
    c_outputs = [state.var_map[var] for var in template.outputs]
    c_trace = _trace_region(c_ops, c_inputs, c_outputs, c_placeholders)
    if t_trace is None or c_trace is None or t_trace != c_trace:
        return False
    # JAX cannot see what a callback computes, so identical traces do not
    # imply identical regions unless the user says so.
    return ctx.assume or 'callback' not in t_trace[0]


def _match(template, c_graph, c_inputs, c_outputs, c_ops, assume_custom_ops_match):
    """How a candidate region computes ``template``'s function: for each template
    input, the position of the matching candidate input; None if they differ.

    The candidate region is ``c_ops`` (all operations of ``c_graph`` if None)
    with inputs ``c_inputs`` and outputs ``c_outputs``, which match the
    template's outputs in order.
    """
    if len(c_inputs) != len(template.inputs) or len(c_outputs) != len(template.outputs):
        return None
    c_input_set = set(c_inputs)
    ctx = _MatchContext(
        template, c_graph,
        input_ok=lambda var: var in c_input_set,
        leaf_ok=lambda var: var not in c_input_set and c_graph.in_degree(var) == 0,
        op_ok=(lambda op: True) if c_ops is None else (lambda op: op in c_ops),
        assume_custom_ops_match=assume_custom_ops_match,
    )
    state = _walk(_MatchState(), list(zip(template.outputs, c_outputs)), ctx)
    state = _complete(state, ctx) if state is not None else None
    if state is None:
        return None
    # Inputs the function reads are mapped exactly; inputs it ignores pair up in order.
    position = {var: j for j, var in enumerate(c_inputs)}
    unused = [j for j, var in enumerate(c_inputs) if var not in state.used_vars]
    permutation = []
    for t_var in template.inputs:
        if t_var in state.var_map:
            permutation.append(position[state.var_map[t_var]])
        elif unused and c_inputs[unused[0]].shape == t_var.shape:
            permutation.append(unused.pop(0))
        else:
            return None
    if unused or not _traces_match(state, ctx, [c_inputs[j] for j in permutation]):
        return None
    return permutation


def _compress_repeats(template, graph, num_inputs):
    """Compress every other copy of ``template``'s region in ``graph`` with its compiled function."""
    import rustworkx as rx
    t_graph = template.get_subgraph()
    t_output = template.outputs[0]
    t_anchor = t_graph.predecessors(t_output)[0]
    port = next(k for k, var in enumerate(t_anchor.outputs) if var is t_output)
    registered = set(template.recorder.constraints) | set(template.recorder.objectives)
    anchors = [graph.rxgraph[i] for i in rx.topological_sort(graph.rxgraph)]
    anchors = [op for op in anchors if type(op) is type(t_anchor) and len(op.outputs) == len(t_anchor.outputs)]

    repeats = []
    for anchor in anchors:
        if anchor not in graph.node_table:  # compressed into an earlier copy
            continue
        ctx = _MatchContext(
            template, graph,
            input_ok=lambda var: True,
            leaf_ok=lambda var: _is_literal(var) and graph.in_degree(var) == 0,
            op_ok=lambda op: op in graph.node_table,
            assume_custom_ops_match=template.assume_custom_ops_match,
        )
        state = _walk(_MatchState(), [(t_output, anchor.outputs[port])], ctx)
        state = _complete(state, ctx) if state is not None else None
        if state is None or any(var not in state.var_map for var in template.inputs):
            continue

        # The same checks compress applies to a region: no intermediate used
        # outside the copy, and no input that depends on the copy itself.
        ops = set(state.op_map.values())
        inputs = [state.var_map[var] for var in template.inputs]
        outputs = [state.var_map[var] for var in template.outputs]
        internal = {var for op in ops for var in op.outputs}
        output_set = set(outputs)
        if internal & set(inputs):
            continue
        escapes = any(
            var in registered or any(user not in ops for user in graph.rxgraph.successors(graph.node_table[var]))
            for var in internal - output_set
        )
        op_indices = {graph.node_table[op] for op in ops}
        if escapes or any(op_indices & rx.ancestors(graph.rxgraph, graph.node_table[var]) for var in inputs):
            continue
        if not _traces_match(state, ctx, inputs):
            continue

        constants = [c for t, c in state.var_map.items() if t not in set(template.inputs) and not t_graph.predecessors(t)]
        region = graph.extract_subgraph_nodes(ops | internal | set(inputs) | set(constants))
        region.name = template.name
        graph._delete_nodes(
            [var for var in internal if var not in output_set]
            + [var for var in constants if graph.out_degree(var) == 0]
        )
        copy = JaxCompressedOperation(
            region, inputs, outputs, name=template.name, device=template.device,
            jit_kwargs=template.jit_kwargs, compile_separately=template.compile_separately,
            share_compiled=False, assume_custom_ops_match=template.assume_custom_ops_match)
        copy.share_compiled = template.share_compiled
        _share_explicitly(copy, template, list(range(len(inputs))))
        copy.extra_inputs = inputs[num_inputs:]
        copy.extra_outputs = outputs[len(outputs) - len(template.extra_outputs):]
        copy.repeats = []
        copy.finalize_and_return_outputs(skip_inline=True)
        repeats.append(copy)
    return repeats


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

    def test_repeated_regions_share_compiled_function(self):
        import csdl_alpha as csdl
        x_vals = [np.array([0.1, 0.4, 0.7]) + 0.2 * k for k in range(3)]
        t_vals = [np.array([1.0 + 0.1 * k]) for k in range(3)]
        c_val = np.array([0.5, -1.0, 2.0])
        f_np = np.array([sum(np.sum(np.sin(x * t)**2 * c_val) for x, t in zip(x_vals, t_vals))])
        for compile_separately in [True, False]:
            self.prep()
            c = csdl.Variable(name='c', value=c_val)
            xs = [csdl.Variable(name=f'x{k}', value=v) for k, v in enumerate(x_vals)]
            ts = [csdl.Variable(name=f't{k}', value=v) for k, v in enumerate(t_vals)]
            rs = [csdl.sum(csdl.sin(x * t)**2 * c) for x, t in zip(xs, ts)]
            op = compress([xs[0], ts[0]], rs[0], find_repeats=True, compile_separately=compile_separately)
            assert len(op.repeats) == 2 and all(r.shared_from is op for r in op.repeats)
            f = rs[0] + rs[1] + rs[2]
            self.run_tests(
                [csdl_tests.TestingPair(f, f_np, tag=f'find_repeats separate={compile_separately}')],
                verify_derivatives=True)

            # The same with explicit sharing and the inputs in another order.
            self.prep()
            c = csdl.Variable(name='c', value=c_val)
            xs = [csdl.Variable(name=f'x{k}', value=v) for k, v in enumerate(x_vals)]
            ts = [csdl.Variable(name=f't{k}', value=v) for k, v in enumerate(t_vals)]
            rs = [csdl.sum(csdl.sin(x * t)**2 * c) for x, t in zip(xs, ts)]
            op = compress([xs[0], ts[0]], rs[0], share_compiled=False, compile_separately=compile_separately)
            for k in (1, 2):
                other = compress([ts[k], xs[k]], rs[k], share_with=op, compile_separately=compile_separately)
                assert other.shared_from is op
            f = rs[0] + rs[1] + rs[2]
            self.run_tests(
                [csdl_tests.TestingPair(f, f_np, tag=f'share_with separate={compile_separately}')],
                verify_derivatives=True)


if __name__ == '__main__':
    test = TestCompressOp()
    test.overwrite_backend = 'jax'
    test.test_simple()