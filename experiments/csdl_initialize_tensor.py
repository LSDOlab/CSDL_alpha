import python_csdl_backend
import csdl


m = csdl.Model()
x = m.create_output('x', shape = (5,10))

a = m.create_input('a', shape = (5,5), val = 2.0)
x[0:5, 0:5] = a

m.register_output('f_x', x*1.0)

b = m.create_input('b', shape = (5,5), val = 3.0)
x[0:5, 5:10] = b

graph = csdl.GraphRepresentation(m)
graph.visualize_graph()

sim = python_csdl_backend.Simulator(graph, analytics=1)
sim.run()
print(sim['x']) #2,3
print(sim['a']) #2
print(sim['b']) #3
print(sim['f_x']) #2,1


