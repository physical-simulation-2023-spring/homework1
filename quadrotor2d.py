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
theta = ti.float32()

# The velocity and the angular velocity of the object.
velocity = tm.vec2()
angular_velocity = ti.float32()
rotation_mat_time_derivative = tm.mat2()

external_force = tm.vec2()
external_torque = ti.float32()

time_step = 0.001
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
    trans.translation = [0, 0.5]
    trans.rotation = [[1., 0.], [0., 1.]]
    
@ti.func
def Forward(h):
    pass

@ti.func
def ForwardEuler():
    trans.translation += velocity * time_step
    theta += angular_velocity * time_step
    trans.UpdateFromTheta(theta)
    velocity += external_force * (time_step / mass)
    angular_velocity += external_torque * (time_step / body_inertia)

@ti.kernel
def ApplyForce(left_delta: float, right_delta: float):
    external_force = gravitational_acceleration * mass
    g = external_force.norm()
    force_direction = trans.rotation[1]
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

while window.running:
    canvas = window.get_canvas()
    canvas.set_background_color((0.1, 0.2, 0.8))
    
    # Draw the robot body.
    line_field = ti.Vector.field(2, dtype=float, shape=(num_links + 1, ))
    # Draw the robot wings.
    canvas.circles(position, ball_radius, (1., 0.7, 0.2))

    ApplyForce()