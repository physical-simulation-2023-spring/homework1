import taichi as ti
import taichi.math as tm
import numpy as np
ti.init(arch=ti.cpu)

num_links = 2
pendulum_length = 2. / num_links
ball_radius = 0.2 * pendulum_length
ball_mass = 25. * ball_radius ** 2 
ball_inertia = 0.5 * ball_mass * ball_radius ** 2

position = ti.Vector.field(2, dtype=float, shape=(num_links, ))
velocity = ti.Vector.field(2, dtype=float, shape=(num_links, ))
angular_velocity = ti.Vector.field(1, dtype=float, shape=(num_links, ))
rotation_matrices = ti.Matrix.field(2, 2, dtype=float, shape=(num_links, ))
external_force_and_torque = ti.Vector([0., -9.8, 0., -9.8, 0., 0.])

generalized_position = ti.Vector([tm.pi / 4, 0.])
generalized_velocity = ti.Vector([0., 0.])
mass_matrix = ti.Matrix(np.diag([ball_mass] * (num_links * 2) + [ball_inertia] * num_links))

jacobian = ti.Matrix([[0.] * num_links] * (num_links * 3))
is_angular_jacobian_initialized = False
jacobian_time_derivative = ti.Matrix([[0.] * num_links] * (num_links * 3))

@ti.func
def get_rotation_mat(q):
    return ti.Matrix([[tm.cos(q), tm.sin(q)], [-tm.sin(q), tm.cos(q)]])

@ti.func
def UpdateLinkPose():
    ti.loop_config(serialize=True)
    for i in rotation_matrices:
        if i == 0:
            rotation_matrices[i] = get_rotation_mat(generalized_position[i])
            position[i] = rotation_matrices[i] @ ti.Vector([0., -pendulum_length])
        else:
            rotation_matrices[i] = rotation_matrices[i - 1] @ get_rotation_mat(generalized_position(i))
            position[i] = rotation_matrices[i] @ ti.Vector([0., -pendulum_length]) + position[i - 1]

@ti.func
def ComputeJacobian():
    # In our case, the angular jacobian is constant.
    if not is_angular_jacobian_initialized:
        for i in range(num_links):
            for j in range(num_links):
                if i <= j:
                    jacobian[num_links * 2 + i, j] = 1
                else:
                    jacobian[num_links * 2 + i, j] = 0
        is_angular_jacobian_initialized = True
    # Then we compute the linear Jacobian.
    ti.loop_config(serialize=True)
    for i in range(num_links):
        for j in range(2):
            for k in range(num_links):
                jacobian[2 * i + j, k] = ...

@ti.func
def ComputeJacobianTimeDerivative():
    pass

@ti.func
def ComputeGeneralizedForce():
    Q = jacobian.transpose() @ external_force_and_torque
    return Q
    
@ti.func
def ComputeCoriolisTerm():
    C = jacobian.transpose() @ mass_matrix @ jacobian_time_derivative @ generalized_velocity
    return C

@ti.func
def ComputeGeneralizedMass():
    M = jacobian.transpose() @ mass_matrix @ jacobian
    return M

@ti.func
def SolveGeneralizedAcceleration(generalized_mass, coriolis_term, generalized_force):
    A = generalized_mass
    b = generalized_force - coriolis_term
    return ti.solve(A, b)

@ti.func
def ForwardEulerTimeIntegration(generalized_acceleration, time_step):
    generalized_position += generalized_velocity * time_step
    generalized_velocity += generalized_acceleration * time_step

@ti.kernel
def step(time_step: float):
    ComputeJacobian()
    ComputeJacobianTimeDerivative()
    Q = ComputeGeneralizedForce()
    C = ComputeCoriolisTerm()
    M = ComputeGeneralizedMass()
    generalized_acceleration = SolveGeneralizedAcceleration(M, C, Q)
    ForwardEulerTimeIntegration(generalized_acceleration, time_step)
    UpdateLinkPose()


@ti.kernel
def render():
    pass

@ti.kernel
def initialize():
    UpdateLinkPose()


print(position[0], rotation_matrices[0])
print(rotation_matrices[0] @ position[0])
position[0] += position[1]
print(position)
input()