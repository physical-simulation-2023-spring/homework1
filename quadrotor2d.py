from core import *

# Simple robot wings.
material_space_robot_wings_coord = ti.Vector.field(2, float, shape=(8, ))
robot_wings_coord = ti.Vector.field(2, float, shape=(8, ))
robot_wings_index = ti.Vector.field(2, int, shape=(4, ))

# Simple robot body.
material_space_robot_body_coord = ti.Vector.field(2, float, shape=(2, ))
robot_body_coord = ti.Vector.field(2, float, shape=(2, ))

# The global transformation of the object.
trans = Transform2D()
theta = ti.field(float, shape=())

# The velocity and the angular velocity of the object.
velocity = ti.Vector.field(2, float, shape=())
angular_velocity = ti.field(float, shape=())

external_force = ti.Vector.field(2, float, shape=())
external_torque = ti.field(float, shape=())

fps = 60
time_step = 0.001
sub_step_num = int(1 / fps / time_step)

mass = 4.
gravitational_acceleration = tm.vec2([0, -9.8])
body_inertia = 0.5

@ti.kernel
def Initialize():
    # We put all data initialization here.
    # Initialize robot wings.
    material_space_robot_wings_coord = [[0.2, 0.1], [0.3, 0.1], [0.25, 0.1], [0.25, 0], [0.7, 0.1], [0.8, 0.1], [0.75, 0.1], [0.75, 0]]
    material_space_robot_body_coord = [[0.25, 0], [0.75, 0]]
    robot_wings_index = [[0, 1], [2, 3], [4, 5], [6, 7]]
    
    # At t0, the quadrotor is at (0, 0.5) with identity rotation.
    trans.translation[None] = [0, 0.5]
    trans.rotation[None] = [[1., 0.], [0., 1.]]
    
@ti.func
def Forward(h):
    pass

@ti.kernel
def ForwardEuler():
    trans.translation[None] += velocity[None] * time_step
    theta[None] += angular_velocity[None] * time_step
    trans.UpdateFromTheta(theta[None])
    velocity[None] += external_force[None] * time_step / mass
    angular_velocity[None] += external_torque[None] * time_step / body_inertia

@ti.kernel
def ApplyForce(left_delta: float, right_delta: float):
    external_force = gravitational_acceleration * mass
    g = external_force.norm()
    force_direction = tm.vec2([trans.rotation[None][1, 0], trans.rotation[None][1, 1]])
    external_force += force_direction * (left_delta + right_delta) * g
    external_torque = (left_delta - right_delta) * g * 0.25

@ti.func
def RungeKutta2():
    pass

@ti.func
def RungeKutta4():
    pass

@ti.kernel
def step():
    pass

control_signal = [
    [0, 0],
    [0, 0],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [1, -1],
    [1, -1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
]

window = ti.ui.Window('Window Title', res = (640, 360), pos = (150, 150))

i = 0
while window.running:
    canvas = window.get_canvas()
    canvas.set_background_color((0.1, 0.2, 0.8))
    
    # Draw the robot body.
    trans.ApplyToPoints(material_space_robot_body_coord, robot_body_coord)
    canvas.lines(robot_body_coord, 0.04, robot_wings_index, color=(0.1, 0.7, 0.3))

    # Draw the robot wings.
    trans.ApplyToPoints(material_space_robot_wings_coord, robot_wings_coord)
    canvas.lines(robot_wings_coord, 0.01, robot_wings_index, (1., 0.7, 0.2))

    for j in range(sub_step_num):
        ApplyForce(*control_signal[i % len(control_signal)])
        ForwardEuler()
    
    i += 1

    window.show()
