from csdl_alpha.src.graph.operation import ComposedOperation

class Sub(ComposedOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'sub'

    def evaluate_composed(self,x,y):
        return evaluate_sub(x,y)


def evaluate_sub(x, y):
    return x+(-y)

def sub(x,y):
    """
    doc strings
    """

    op = Sub(x,y)
    return op.get_outputs()

# def sub(x,y):
#     """
#     doc strings
#     """

#     return evaluate_sub(x,y)