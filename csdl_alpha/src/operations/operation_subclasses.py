from csdl_alpha.src.graph.operation import Operation, set_properties


@set_properties(elementwise = True, diagonal_jacobian = True)
class ElementwiseOperation(Operation):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        out_shapes = (args[0].shape,)
        self.set_dense_outputs(out_shapes)

@set_properties(elementary = False)
class ComposedOperation(Operation):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.recorder._enter_subgraph()
        
        for input in self.inputs:
            self.recorder._add_node(input)
        outputs = self.evaluate_composed(*args)

        if isinstance(outputs, tuple):
            self.set_outputs(outputs)
        else:
            self.set_outputs([outputs])

        # self.recorder.active_graph.link_parent_operation(self)
        self.subgraph = self.recorder.active_graph

        self.recorder._exit_subgraph()

        for output in self.outputs:
            self.recorder._add_node(output)

    def finalize_and_return_outputs(self):
        outputs = super().finalize_and_return_outputs(skip_inline = True)
        return outputs
    
    def set_inline_values(self):
        self.subgraph.execute_inline()


# def set_property(cls, name, value):
#     properties = cls.properties.copy()
#     properties[name] = value
#     cls.properties = properties

# class ElementwiseOperation(Operation):
#     properties = Operation.properties.copy()
#     # properties['elementwise'] = True
#     # set_property(ElementwiseOperation, 'elementwise', True)
#     # # properties = ReadOnlyDict(properties)

#     def __init__(self,*args, **kwargs):
        
#         super().__init__(*args, **kwargs)

#         self.properties['elementwise'] = True
#         self.properties['diagonal_jacobian'] = True

#         out_shape = (args[0].shape,)
#         self.set_dense_outputs(out_shape)

# set_property(ElementwiseOperation, 'elementwise', True)
# # ElementwiseOperation.properties = Operation.properties.copy()
# # ElementwiseOperation.properties['elementwise'] = True

# class ComposedOperation(Operation):

#     def __init__(self,*args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.properties['elementary'] = False

#         self.recorder._enter_subgraph()
        
#         for input in self.inputs:
#             self.recorder._add_node(input)
#         outputs = self.evaluate_composed(*args)

#         if isinstance(outputs, tuple):
#             self.outputs = outputs
#         else:
#             self.outputs = [outputs]
#         self.graph = self.recorder.active_graph

#         self.recorder._exit_subgraph()

#         for output in self.outputs:
#             self.recorder._add_node(output)

#     def get_outputs(self):
#         outputs = super().get_outputs()
#         return outputs

    
#     def set_inline_values(self):
#         pass