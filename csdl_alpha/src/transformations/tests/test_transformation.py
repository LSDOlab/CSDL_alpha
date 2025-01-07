import csdl_alpha as csdl

def f(x1, x2):
    z = x1*x2
    A = csdl.expand(x1, (3,2))
    b = csdl.expand(x2, (2,3))
    d = csdl.sin(z)*A
    return d, A@b

class TestTransformation(csdl.transforms.TransformationBase):

    def post_init(self):
        self.metadata = {}

    def apply(self):
        recorder = self.get_current_recorder()
        graph = recorder.active_graph

        ct = csdl.transformation_helper

        del_nodes = []
        for i, node in enumerate(graph.node_table):
            if not ct.is_var(node):
                del_nodes.append(node)
        ct.delete_nodes(del_nodes)

def test_transform():

    recorder =csdl.Recorder(inline = True)
    recorder.start()
    x1 = csdl.Variable(value = 1.0)
    x2 = csdl.Variable(value = 2.0)

    f1, f2 = f(x1, x2)
    print(f1.value)
    print(f2.value)

    da_transform = TestTransformation(name = 'oo')
    da_transform.transform()

    print()
    f1.check_history()
    print()
    recorder.transformation_logger.print_history()
    recorder.transformation_logger.print_stack()
    print()
    print(da_transform.info())
    print(recorder.transformation_logger.get_current().info())
    
if __name__ == '__main__':
    test_transform()