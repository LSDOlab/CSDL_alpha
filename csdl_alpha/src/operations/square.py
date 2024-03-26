from csdl_alpha.src.graph.operation import ComposedOperation

class Square(ComposedOperation):

    def __init__(self,x):
        super().__init__(x)
        self.name = 'sqr'

    def evaluate_composed(self,x):
        return evaluate_square(x)


def evaluate_square(x):
    return x*x

def square(x):
    """
    doc strings
    """

    op = Square(x)
    return op.get_outputs()
