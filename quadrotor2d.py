from core import *

# Simple robot propeller.
material_space_robot_propeller_coord = ti.Vector.field(2, float, shape=(8, ))
for i, data in enumerate([[0.3, 0.1], [0.4, 0.1], [0.35, 0.1], [0.35, 0], [0.6, 0.1], [0.7, 0.1], [0.65, 0.1], [0.65, 0]]):
    material_space_robot_propeller_coord[i] = data
robot_propeller_coord = ti.Vector.field(2, float, shape=(8, ))
robot_propeller_index = ti.Vector.field(2, int, shape=(4, ))
for i, data in enumerate([[0, 1], [2, 3], [4, 5], [6, 7]]):
    robot_propeller_index[i] = data

# Simple robot body.
material_space_robot_body_coord = ti.Vector.field(2, float, shape=(2, ))
material_space_robot_body_coord[0] = [0.35, 0]
material_space_robot_body_coord[1] = [0.65, 0]
robot_body_coord = ti.Vector.field(2, float, shape=(2, ))

# The global transformation of the object.
trans = Transform2D()
trans.translation[None] = [0, 0.5]
trans.rotation[None] = [[1., 0.], [0., 1.]]
theta = ti.field(float, shape=())

# The velocity and the angular velocity of the object.
velocity = ti.Vector.field(2, float, shape=())
angular_velocity = ti.field(float, shape=())

external_force = ti.Vector.field(2, float, shape=())
external_torque = ti.field(float, shape=())

fps = 300
time_step = 0.0001
sub_step_num = int(1 / fps / time_step)

mass = 4.
gravitational_acceleration = tm.vec2([0, -9.8])
body_inertia = 0.5
    
@ti.func
def Forward(h):
    pass

@ti.kernel
def ForwardEuler():
    # print("FF:", external_force[None], external_torque[None])
    # input("Finish print.")
    trans.translation[None] += velocity[None] * time_step
    theta[None] += angular_velocity[None] * time_step
    trans.UpdateFromTheta(theta[None])
    velocity[None] += external_force[None] * time_step / mass
    angular_velocity[None] += external_torque[None] * time_step / body_inertia
    # print("FF, rot, theta:", theta[None], trans.rotation[None])

@ti.kernel
def ApplyForce(left_delta: float, right_delta: float):
    external_force[None] = gravitational_acceleration * mass
    g = external_force[None].norm()
    force_direction = tm.vec2([trans.rotation[None][1, 0], trans.rotation[None][1, 1]])
    external_force[None] += force_direction * (left_delta + right_delta + 1) * g
    external_torque[None] = (left_delta - right_delta) * g * 0.25
    # print("AP:", external_force[None], external_torque[None])
    # print("AP, rot:", trans.rotation[None])

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
    [.01, .01],
    [.01, .01],
    [.01, .01],
    [.01, .01],
    [-.01, -.01],
    [-.01, -.01],
    [-.01, -.01],
    [.01, -.01],
    [-.01, .01],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
]


i = 0

# for j in range(sub_step_num):
#     ApplyForce(*control_signal[i % len(control_signal)])
#     ForwardEuler()
#     input("mid")

# input("Fin.")

window = ti.ui.Window('Window Title', res = (640, 360), pos = (150, 150))
canvas = window.get_canvas()
canvas.set_background_color((0.1, 0.1, 0.1))

# trans.ApplyToPoints(material_space_robot_body_coord, robot_body_coord)
# # while window.running:
# #     canvas.lines(robot_body_coord, 0.04, color=(0.1, 0.7, 0.3))
# #     window.show()

# Test for 
while window.running:
    # Draw the robot body.
    trans.ApplyToPoints(material_space_robot_body_coord, robot_body_coord)
    canvas.lines(robot_body_coord, 0.04, color=(0.1, 0.7, 0.3))

    # Draw the robot propeller.
    trans.ApplyToPoints(material_space_robot_propeller_coord, robot_propeller_coord)
    canvas.lines(robot_propeller_coord, 0.01, robot_propeller_index, (1., 0.7, 0.2))

    for j in range(sub_step_num):
        ApplyForce(*control_signal[i % len(control_signal)])
        ForwardEuler()
    
    i += 1

    window.show()
