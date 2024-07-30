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

if __name__ == '__main__':
    test = TestCompressOp()
    test.overwrite_backend = 'jax'
    test.test_simple()