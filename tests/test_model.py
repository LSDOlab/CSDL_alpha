import pytest

def define_my_model():
    import csdl_alpha as csdl
    class MyModel(csdl.Model):
        def initialize(self):
            a = self.parameters.declare('a', types = float)
            self.a2 = a**2
        def evaluate(self):
            b = csdl.Variable((1,), name='b', value=self.a2)
            return b
    return MyModel

def test_model_correct():
    import csdl_alpha as csdl
    MyModel = define_my_model()
        
    recorder = csdl.build_new_recorder()
    recorder.start()
    model = MyModel(a=3.)
    b = model.evaluate(name='test')
    b2 = model.evaluate()
    recorder.stop()
    assert b.namespace.name == 'test'
    assert b.value == 9
    assert b2.namespace.name == 'MyModel'
    assert b2.value == 9

def test_wrong_inputs():
    import csdl_alpha as csdl
    MyModel = define_my_model()
        
    recorder = csdl.build_new_recorder()
    recorder.start()
    with pytest.raises(Exception) as e_info:
        model = MyModel(a=3., b=3.)


test_wrong_inputs()
test_model_correct()