import taichi as ti
import taichi.math as tm
import numpy as np
ti.init(arch=ti.cpu)

num_links = 2
pendulum_length = 1.
ball_radius = 0.2
ball_mass = 1.
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
jacobian_time_derivative = ti.Matrix([[0.] * num_links] * (num_links * 3))

@ti.func
def UpdateLinkPose():
    for i in rotation_matrices:
        pass

@ti.func
def ComputeJacobian():
    pass

@ti.func
def ComputeJacobianTimeDerivative():
    pass

@ti.func
def ComputeGeneralizedForce():
    Q = jacobian.T @ external_force_and_torque
    return Q
    
@ti.func
def ComputeCoriolisTerm():
    C = jacobian.T @ mass_matrix @ jacobian_time_derivative @ generalized_velocity
    return C

@ti.func
def ComputeGeneralizedMass():
    M = jacobian.T @ mass_matrix @ jacobian
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