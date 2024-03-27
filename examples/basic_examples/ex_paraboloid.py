if __name__ == '__main__':
    import csdl_alpha as csdl
    import numpy as np

    class ParaboloidModel(csdl.Model):

        def initialize(self):
            self.a = self.parameters.declare('a')
            self.b = self.parameters.declare('b')
            self.c = self.parameters.declare('c')

        def evaluate(self, x, y):
            # x and y are the same python objects as used in line 19
            f = csdl.square(x-self.a) + x*y + csdl.square(y+self.b) - self.c

            # Optionally set a variable name
            f.add_name('f') # This gives f the name 'paraboloid.f'
            return f

    recorder = csdl.build_new_recorder(inline=True, debug=True, auto_hierarchy=True)
    recorder.start()


    csdl.enter_namespace('test')
    x = csdl.Variable(value=2., name='x')
    csdl.exit_namespace()
    x.save = True
    x.add_tag('banana')
    x.add_tag('apple')
    y = csdl.Variable(value=1., name='y')

    a = 12
    b = 5
    c = 30
    # Add a sub-model to the current model
    parabolid_model = ParaboloidModel(a=a, b=b, c=c)

    # Call the evaluate method of the model functionally.
    f = parabolid_model.evaluate(x, y, name = 'paraboloid_submodel')
    f.save = True

    recorder.active_graph.visualize()

    assert f.value == np.array([108])
    # f.print_trace()
    csdl.save_all_variables()
    csdl.inline_save('test')
    # variables = csdl.import_h5py('test.hdf5', 'inline')
    

    recorder.stop()