'''
Cantilever Beam model:
'''
import numpy as np
import csdl_alpha as csdl
from csdl_alpha.src.operations.linalg.vdot import vdot



# cantelever beam model based on OpenMDAO example beam 
# https://github.com/OpenMDAO/OpenMDAO/blob/master/openmdao/test_suite/test_examples/beam_optimization/beam_group.py
class BeamModel():

    def __init__(self, E, L, b, num_elements:int):
        self.parameters = {'E': E, 'L': L, 'b': b, 'num_elements': num_elements}

    def evaluate(self, force_vector, h):
        E = self.parameters['E']
        L = self.parameters['L']
        b = self.parameters['b']
        num_elements = self.parameters['num_elements']

        moment_of_inertia_comp = MomentOfInertiaComp(b=b)

        with csdl.Namespace('inertia'):
            I = moment_of_inertia_comp.evaluate(h=h)

        local_stiffness_matrix_comp = LocalStiffnessMatrixComp(num_elements=num_elements, E=E, L=L)
        with csdl.Namespace('stiffness'):
            K_local = local_stiffness_matrix_comp.evaluate(I=I)

        states_comp = StatesComp(num_elements=num_elements)
        with csdl.Namespace('states'):
            d = states_comp.evaluate(K_local=K_local, force_vector=force_vector)

        compliance_comp = ComplianceComp()
        with csdl.Namespace('compliance'):
            compliance = compliance_comp.evaluate(d=d[:-2], force_vector=force_vector)

        volume_comp = VolumeComp(num_elements=num_elements, L=L)
        with csdl.Namespace('volume'):
            volume = volume_comp.evaluate(h=h, b=b)

        return d, compliance, volume
    
class MomentOfInertiaComp():

    def __init__(self, b):
        self.parameters = {'b': b}

    def evaluate(self, h):
        I = 1./12. * self.parameters['b'] * h**3
        return I
    
class LocalStiffnessMatrixComp():

    def __init__(self, num_elements:int, E, L):
        self.parameters = {'num_elements': num_elements, 'E': E, 'L': L}

        L0 = L / num_elements
        coeffs = np.empty((4, 4))
        coeffs[0, :] = [12, 6 * L0, -12, 6 * L0]
        coeffs[1, :] = [6 * L0, 4 * L0 ** 2, -6 * L0, 2 * L0 ** 2]
        coeffs[2, :] = [-12, -6 * L0, 12, -6 * L0]
        coeffs[3, :] = [6 * L0, 2 * L0 ** 2, -6 * L0, 4 * L0 ** 2]
        coeffs *= E / L0 ** 3

        self.coeffs = coeffs

    def evaluate(self, I):
        num_elements = self.parameters['num_elements']
        coeffs = self.coeffs

        mtx = np.zeros((num_elements, 4, 4, num_elements))
        for ind in range(num_elements):
            mtx[ind, :, :, ind] = coeffs

        K_local = csdl.Variable(value=np.zeros((num_elements, 4, 4)))
        mtx = csdl.Variable(value=mtx)

        for ind in csdl.frange(num_elements):
            K_local = K_local.set(csdl.slice[ind, :, :], mtx[ind, :, :, ind] * I[ind])
            
        return K_local
    
class StatesComp():

    def __init__(self, num_elements:int):
        self.parameters = {'num_elements': num_elements}

    def evaluate(self, K_local, force_vector):
        K = self.assemble_CSC_K(K_local)
        force_vector_init = csdl.Variable(value=np.concatenate([np.zeros(force_vector.shape), np.zeros(2)]))
        force_vector = force_vector_init.set(csdl.slice[:-2], force_vector)
        print(force_vector.shape, K.shape)
        d = csdl.solve_linear(K, force_vector)   # solve operations should be flexible for FS/RS
        print(d.shape)
        return d


    def assemble_CSC_K(self, K_local):
        """
        Assemble the stiffness matrix in sparse CSC format.

        Returns
        -------
        csdl.csc array
            Stiffness matrix as sparse csdl var.
        """
        num_elements = self.parameters['num_elements']
        num_nodes = num_elements + 1
        num_entry = num_elements * 12 + 4
        ndim = num_entry + 4

        data = csdl.Variable(value=np.zeros((ndim, )))
        cols = np.empty((ndim, ))
        rows = np.empty((ndim, ))

        # First element.
        data = data.set(csdl.slice[:16], K_local[0, :, :].flatten())
        cols[:16] = np.tile(np.arange(4), 4)
        rows[:16] = np.repeat(np.arange(4), 4)

        j = j_offset = 16
        for ind in range(1, num_elements): 
            ind1 = 2 * ind
            # NE quadrant
            rows[j:j+4] = np.array([ind1, ind1, ind1 + 1, ind1 + 1])
            cols[j:j+4] = np.array([ind1 + 2, ind1 + 3, ind1 + 2, ind1 + 3])

            # SE and SW quadrants together
            rows[j+4:j+12] = np.repeat(np.arange(ind1 + 2, ind1 + 4), 4)
            cols[j+4:j+12] = np.tile(np.arange(ind1, ind1 + 4), 2)

            j += 12

        for ind in csdl.frange(1, num_elements):
            j = j_offset + (ind-1) * 12
            K = K_local[ind, :, :]

            # NW quadrant gets summed with previous connected element.
            indices1 = [j-6, j-5]
            indices2 = [j-2, j-1]
            data = data.set(csdl.slice[indices1], data[indices1] + K[0, :2])
            data = data.set(csdl.slice[indices2], data[indices2] + K[1, :2])

            # NE quadrant
            data = data.set(csdl.slice[j:j+4], K[:2, 2:].flatten())

            # SE and SW quadrants together
            data = data.set(csdl.slice[j+4:j+12], K[2:, :].flatten())

        data = data.set(csdl.slice[-4:], 1.0)
        rows[-4] = 2 * num_nodes
        rows[-3] = 2 * num_nodes + 1
        rows[-2] = 0.0
        rows[-1] = 1.0
        cols[-4] = 0.0
        cols[-3] = 1.0
        cols[-2] = 2 * num_nodes
        cols[-1] = 2 * num_nodes + 1

        n_K = 2 * num_nodes + 2

        # ready for sparse, but turning to dense for now
        K_global = csdl.Variable(value=np.zeros((n_K, n_K)))
        K_global = K_global.set(csdl.slice[list(rows.astype(int)), list(cols.astype(int))], data)
                            
        return K_global
    

class ComplianceComp():

    def evaluate(self, d, force_vector):
        compliance = vdot(force_vector, d)
        return compliance
    
class VolumeComp():

    def __init__(self, num_elements:int, L):
        self.parameters = {'num_elements': num_elements, 'L': L}

    def evaluate(self, h, b):
        L0 = self.parameters['L'] / self.parameters['num_elements']

        volume = csdl.sum(h * b * L0)
        return volume
    



recorder = csdl.Recorder(inline=True)
recorder.start()

E = 1.
L = 1.
b = 0.1
num_elements = 1000
force_vector = np.zeros(2 * (num_elements + 1))
force_vector[-2] = -1.
h = csdl.Variable(value=np.ones(num_elements) * 0.5)
# h = np.array(
# [0.14915751, 0.14764323, 0.14611341, 0.14456713, 0.14300423, 0.14142421,
#  0.13982606, 0.13820962, 0.13657403, 0.13491857, 0.13324265, 0.1315453,
#  0.12982572, 0.12808315, 0.12631656, 0.12452484, 0.122707,   0.12086183,
#  0.11898806, 0.11708424, 0.11514905, 0.11318069, 0.11117766, 0.10913768,
#  0.10705894, 0.10493899, 0.1027754,  0.10056525, 0.09830549, 0.09599247,
#  0.09362247, 0.0911908,  0.08869256, 0.08612201, 0.08347219, 0.08073579,
#  0.07790323, 0.07496376, 0.07190454, 0.06870931, 0.0653583,  0.06182638,
#  0.05808046, 0.05407658, 0.04975292, 0.04501854, 0.03972909, 0.03363155,
#  0.0262019,  0.01610862]
#  )


beam_model = BeamModel(E, L, b, num_elements)
d, compliance, volume = beam_model.evaluate(force_vector, h)
print(d.value, compliance.value, volume.value)
recorder.stop()
recorder.visualize_graph('beam_graph')
