from core import *

# Simple robot propeller.
material_space_robot_propeller_coord = ti.Vector.field(2, float, shape=(8, ))
robot_propeller_coord = ti.Vector.field(2, float, shape=(8, ))
robot_propeller_index = ti.Vector.field(2, int, shape=(4, ))
for i, data in enumerate([[0.35, 0.1], [0.45, 0.1], [0.4, 0.1], [0.4, 0], [0.55, 0.1], [0.65, 0.1], [0.6, 0.1], [0.6, 0]]):
    material_space_robot_propeller_coord[i] = data
for i, data in enumerate([[0, 1], [2, 3], [4, 5], [6, 7]]):
    robot_propeller_index[i] = data

# Simple robot body.
material_space_robot_body_coord = ti.Vector.field(2, float, shape=(2, ))
material_space_robot_body_coord[0] = [0.4, 0]
material_space_robot_body_coord[1] = [0.6, 0]
robot_body_coord = ti.Vector.field(2, float, shape=(2, ))

# The global transformation of the object.
trans = Transform2D()
theta = ti.field(float, shape=())

# The velocity and the angular velocity of the object.
velocity = ti.Vector.field(2, float, shape=())
angular_velocity = ti.field(float, shape=())

var_name = ["trans", "theta", "velocity", "angular_velocity"]

external_force = ti.Vector.field(2, float, shape=())
external_torque = ti.field(float, shape=())

fps = 60
time_step = 0.001
sub_step_num = int(1 / fps / time_step)

mass = 4.
gravitational_acceleration = tm.vec2([0, -9.8])
body_inertia = 0.5

goal = ti.Vector.field(2, float, shape=(1, ))

force_visualization_coord = ti.Vector.field(2, float, shape=(8, ))
force_visualization_index = ti.Vector.field(2, int, shape=(6, ))
for i, data in enumerate([[0, 1], [1, 2], [1, 3], [4, 5], [5, 6], [5, 7]]):
    force_visualization_index[i] = data

@ti.func
def F(var_q):
    # Newton-Euler equation: compute time derivative of var_q.
    var_q_dot = {var: ti.zero(var_q[var]) for var in var_name}
    var_q_dot["trans"] = var_q["velocity"]
    var_q_dot["theta"] = var_q["angular_velocity"]
    var_q_dot["velocity"] = external_force[None] / mass
    var_q_dot["angular_velocity"] = external_torque[None] / body_inertia
    return var_q_dot
    
# You may want to define your own variables or functions here.
# -- YOUR CODE BEGINS --
# TODO: Visualize the force.
# F: q -> q_dot
# Script = ...
# A set of input and output ...
# 

# R + h dR; R svd -> USV -> UV
# R @ Rod(w * delta t)

# -- END OF YOUR CODE --

@ti.kernel
def Initialize():
    trans.translation[None] = [0, 0.5]
    trans.rotation[None] = [[1., 0.], [0., 1.]]
    velocity[None] = [0, 0]
    angular_velocity[None] = 0
    theta[None] = 0
    goal[0] = [0.5, 0.5]
    # You may want to initialize your variables here.
    # -- YOUR CODE BEGINS --


    # -- END OF YOUR CODE --

@ti.kernel
def ForwardEuler():
    var_q = {"trans": trans.translation[None], "theta": theta[None], 
             "velocity": velocity[None], "angular_velocity": angular_velocity[None]}
    var_q_dot = F(var_q)
    trans.translation[None] += var_q_dot["trans"] * time_step
    theta[None] += var_q_dot["theta"] * time_step
    trans.UpdateRotationFromTheta(theta[None])
    velocity[None] += var_q_dot["velocity"] * time_step
    angular_velocity[None] += var_q_dot["angular_velocity"] * time_step

@ti.kernel
def RungeKutta2():
    var_q = {"trans": trans.translation[None], "theta": theta[None], 
             "velocity": velocity[None], "angular_velocity": angular_velocity[None]}
    var_q_dot_half = F(var_q)
    h_half = time_step / 2
    var_q_half = {
        "trans": var_q["trans"] + var_q_dot_half["trans"] * h_half, 
        "theta": var_q["theta"] + var_q_dot_half["theta"] * h_half,  
        "velocity": var_q["velocity"] + var_q_dot_half["velocity"] * h_half, 
        "angular_velocity": var_q["angular_velocity"] + var_q_dot_half["angular_velocity"] * h_half
    }
    var_q_dot = F(var_q_half)
    trans.translation[None] += var_q_dot["trans"] * time_step
    theta[None] += var_q_dot["theta"] * time_step
    trans.UpdateRotationFromTheta(theta[None])
    velocity[None] += var_q_dot["velocity"] * time_step
    angular_velocity[None] += var_q_dot["angular_velocity"] * time_step    

# External forces.
@ti.kernel
def ApplyForce(left_delta: float, right_delta: float):
    external_force[None] = gravitational_acceleration * mass
    g = external_force[None].norm()
    force_direction = trans.rotation[None] @ tm.vec2([0, 1])
    external_force[None] += force_direction * (left_delta + right_delta + 1) * g
    external_torque[None] = (left_delta - right_delta) * g * 0.25

@ti.kernel
def VisualizeForce(left_delta: float, right_delta: float):
    force_direction = trans.rotation[None] @ tm.vec2([0, 1])
    arrow_left_direction = trans.rotation[None] @ tm.vec2([-0.3, -1])
    arrow_right_direction = trans.rotation[None] @ tm.vec2([0.3, -1])
    force_left_loc = robot_propeller_coord[2]
    force_right_loc = robot_propeller_coord[6]
    force_visualization_coord[0] = force_left_loc + force_direction * 0.03
    force_visualization_coord[1] = force_left_loc + force_direction * (0.13 + left_delta * 0.4)
    force_visualization_coord[2] = force_visualization_coord[1] + arrow_left_direction * 0.03
    force_visualization_coord[3] = force_visualization_coord[1] + arrow_right_direction * 0.03
    force_visualization_coord[4] = force_right_loc + force_direction * 0.03
    force_visualization_coord[5] = force_right_loc + force_direction * (0.13 + right_delta * 0.4)
    force_visualization_coord[6] = force_visualization_coord[5] + arrow_left_direction * 0.03
    force_visualization_coord[7] = force_visualization_coord[5] + arrow_right_direction * 0.03


# PD Control.
@ti.kernel
def YControllor(y_goal: float) -> float:
    # PD control for y.
    target = clamp(y_goal, 0.1, 0.9)
    y = trans.translation[None][1]
    dot_y = velocity[None][1]
    return - 15 * (y - target) - 15 * dot_y

@ti.kernel
def ThetaControllor(theta_goal: float) -> float:
    # PD control for theta.
    target = clamp(theta_goal, -tm.pi / 30, tm.pi / 30)
    return - 9 * (theta[None] - target) - 15 * angular_velocity[None]

@ti.kernel
def XControllor(x_goal: float) -> float:
    # X controllor is a 2nd-level controllor.
    target = clamp(x_goal, -0.4, 0.4)
    x = trans.translation[None][0]
    dot_x = velocity[None][0]
    return 5 * (x - target) + 15 * dot_x - theta[None]

i = 0
time_integrate_method = ["Forward Euler", "RK-2"]
time_integrate_method_index = 0

window = ti.ui.Window('Quadrotor 2D', res = (640, 360), pos = (150, 150), vsync=True)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((0.25, 0.25, 0.25))

Initialize()

while window.running:
    # gui.gui.arrow([0.1, 0.1], [1, 0])
    if window.get_event(ti.ui.PRESS):
        if i > 10:
            if window.event.key == 'r': 
                Initialize()
            elif window.event.key in [ti.ui.ESCAPE]: 
                break
            if window.is_pressed(ti.ui.LEFT, 'a'):
                goal[0][0] -= .05 if goal[0][0] > 0.1 else 0
            if window.is_pressed(ti.ui.RIGHT, 'd'):
                goal[0][0] += .05 if goal[0][0] < 0.9 else 0
            if window.is_pressed(ti.ui.UP, 'w'):
                goal[0][1] += .1 if goal[0][1] < 0.85 else 0
            if window.is_pressed(ti.ui.DOWN, 's'):
                goal[0][1] -= .1 if goal[0][1] > 0.15 else 0
            if window.is_pressed(ti.ui.SHIFT):
                time_integrate_method_index = 1 - time_integrate_method_index
                print(f"Change to {time_integrate_method[time_integrate_method_index]} time integration.")
            i = 0

    # Draw the robot body.
    trans.ApplyToPoints(material_space_robot_body_coord, robot_body_coord)
    canvas.lines(robot_body_coord, 0.04, color=(0.1, 0.7, 0.3))

    # Draw the robot propeller.
    trans.ApplyToPoints(material_space_robot_propeller_coord, robot_propeller_coord)
    canvas.lines(robot_propeller_coord, 0.01, robot_propeller_index, (1., 0.7, 0.2))
    
    # Draw the target point.
    canvas.circles(goal, 0.01, color=(1., 0.1, 0.1))

    for j in range(sub_step_num):
        delta = YControllor(goal[0][1])
        theta_delta = ThetaControllor(XControllor(goal[0][0] - 0.5))
        ApplyForce(delta + theta_delta, delta - theta_delta)
        VisualizeForce(delta + theta_delta, delta - theta_delta)
        if time_integrate_method_index == 0:
            ForwardEuler()
        elif time_integrate_method_index == 1:
            RungeKutta2()
    
    canvas.lines(force_visualization_coord, 0.006, force_visualization_index, (1., 0.1, 0.1))
    i += 1

    window.show()

# visualize the force