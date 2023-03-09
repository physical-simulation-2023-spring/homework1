from release.util import *

# Simple robot propeller.
material_space_robot_propeller_vertex = ti.Vector.field(2, float, shape=(8, ))
robot_propeller_vertex = ti.Vector.field(2, float, shape=(8, ))
robot_propeller_index = ti.Vector.field(2, int, shape=(4, ))
FillData(material_space_robot_propeller_vertex, 
         [[0.35, 0.1], [0.45, 0.1], [0.4, 0.1], [0.4, 0], [0.55, 0.1], [0.65, 0.1], [0.6, 0.1], [0.6, 0]])
FillData(robot_propeller_index, [[0, 1], [2, 3], [4, 5], [6, 7]])

# Simple robot body.
material_space_robot_body_vertex = ti.Vector.field(2, float, shape=(2, ))
material_space_robot_body_vertex[0] = [0.4, 0]
material_space_robot_body_vertex[1] = [0.6, 0]
robot_body_vertex = ti.Vector.field(2, float, shape=(2, ))

# The global transformation of the object.
transform = Transform2D()
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
gravity = (mass * gravitational_acceleration).norm()
body_inertia = 0.5

goal = ti.Vector.field(2, float, shape=(1, ))

force_visualization_vertex = ti.Vector.field(2, float, shape=(8, ))
force_visualization_index = ti.Vector.field(2, int, shape=(6, ))
FillData(force_visualization_index, [[0, 1], [1, 2], [1, 3], [4, 5], [5, 6], [5, 7]])

@ti.kernel
def Initialize():
    transform.translation[None] = [0, 0.5]
    transform.rotation[None] = [[1., 0.], [0., 1.]]
    velocity[None] = [0, 0]
    angular_velocity[None] = 0
    theta[None] = 0
    goal[0] = [0.5, 0.5]

@ti.func
def F(q):
    # Newton-Euler equation: compute time derivative of q.
    q_dot = {var: ti.zero(q[var]) for var in q}
    q_dot["translation"] = q["velocity"]
    q_dot["theta"] = q["angular_velocity"]
    q_dot["velocity"] = external_force[None] / mass
    q_dot["angular_velocity"] = external_torque[None] / body_inertia
    return q_dot

@ti.func
def ForwardEuler():
    q = {"translation": transform.translation[None], "theta": theta[None], 
             "velocity": velocity[None], "angular_velocity": angular_velocity[None]}
    q_dot = F(q)
    transform.translation[None] += q_dot["translation"] * time_step
    theta[None] += q_dot["theta"] * time_step
    transform.UpdateRotationFromTheta(theta[None])
    velocity[None] += q_dot["velocity"] * time_step
    angular_velocity[None] += q_dot["angular_velocity"] * time_step

@ti.func
def RungeKutta2():
    q = {"translation": transform.translation[None], "theta": theta[None], 
             "velocity": velocity[None], "angular_velocity": angular_velocity[None]}
    q_dot_half = F(q)
    h_half = time_step / 2
    q_half = {
        "translation": q["translation"] + q_dot_half["translation"] * h_half, 
        "theta": q["theta"] + q_dot_half["theta"] * h_half,  
        "velocity": q["velocity"] + q_dot_half["velocity"] * h_half, 
        "angular_velocity": q["angular_velocity"] + q_dot_half["angular_velocity"] * h_half
    }
    q_dot = F(q_half)
    transform.translation[None] += q_dot["translation"] * time_step
    theta[None] += q_dot["theta"] * time_step
    transform.UpdateRotationFromTheta(theta[None])
    velocity[None] += q_dot["velocity"] * time_step
    angular_velocity[None] += q_dot["angular_velocity"] * time_step    

@ti.func
def ApplyForce(left_thrust_input: float, right_thrust_input: float):
    left_thrust = clamp(left_thrust_input, 10, 30)
    right_thrust = clamp(right_thrust_input, 10, 30)
    external_force[None] = gravitational_acceleration * mass
    force_direction = transform.rotation[None] @ tm.vec2([0, 1])
    external_force[None] += force_direction * (left_thrust + right_thrust)
    external_torque[None] = (- left_thrust + right_thrust) * 0.25

@ti.kernel
def VisualizeForce(left_thrust_input: float, right_thrust_input: float):
    left_thrust = clamp(left_thrust_input, 10, 30)
    right_thrust = clamp(right_thrust_input, 10, 30)
    force_direction = transform.rotation[None] @ tm.vec2([0, 1])
    arrow_left_direction = transform.rotation[None] @ tm.vec2([-0.3, -1])
    arrow_right_direction = transform.rotation[None] @ tm.vec2([0.3, -1])
    force_left_loc = robot_propeller_vertex[2]
    force_right_loc = robot_propeller_vertex[6]
    force_visualization_vertex[0] = force_left_loc + force_direction * 0.03
    force_visualization_vertex[1] = force_left_loc + force_direction * (left_thrust - 9) * 0.015
    force_visualization_vertex[2] = force_visualization_vertex[1] + arrow_left_direction * 0.03
    force_visualization_vertex[3] = force_visualization_vertex[1] + arrow_right_direction * 0.03
    force_visualization_vertex[4] = force_right_loc + force_direction * 0.03
    force_visualization_vertex[5] = force_right_loc + force_direction * (right_thrust - 9) * 0.015
    force_visualization_vertex[6] = force_visualization_vertex[5] + arrow_left_direction * 0.03
    force_visualization_vertex[7] = force_visualization_vertex[5] + arrow_right_direction * 0.03

# PD Control.
@ti.func
def YControllor(y_goal: float) -> float:
    # PD control for y.
    target = clamp(y_goal, 0, 1)
    y = transform.translation[None][1]
    dot_y = velocity[None][1]
    return (- 15 * (y - target) - 15 * dot_y + 0.5) * gravity

@ti.func
def ThetaControllor(theta_goal: float) -> float:
    # PD control for theta.
    target = clamp(theta_goal, -tm.pi / 15, tm.pi / 15)
    return (9 * (theta[None] - target) + 15 * angular_velocity[None]) * gravity

@ti.func
def XControllor(x_goal: float) -> float:
    # X controllor is a 2nd-level controllor.
    target = clamp(x_goal, -0.5, 0.5)
    x = transform.translation[None][0]
    dot_x = velocity[None][0]
    return 5 * (x - target) + 15 * dot_x - theta[None]

@ti.kernel
def substep(use_RK2: bool) -> tm.vec2:
    main_thrust = YControllor(goal[0][1])
    thrust_delta = ThetaControllor(XControllor(goal[0][0] - 0.5))
    ApplyForce(main_thrust + thrust_delta, main_thrust - thrust_delta)
    if use_RK2:
        RungeKutta2()
    else:
        ForwardEuler()
    return tm.vec2([main_thrust + thrust_delta, main_thrust - thrust_delta])

i = 0
time_integrate_method = "Forward Euler"
verbose = False

window = ti.ui.Window('Quadrotor 2D', res = (640, 360), pos = (600, 350), vsync=True)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((0.15, 0.15, 0.15))

Initialize()

while window.running:
    if window.get_event(ti.ui.PRESS):
        if i > 10:
            if window.event.key == 'r': 
                Initialize()
            elif window.event.key in [ti.ui.ESCAPE]: 
                break
            if window.is_pressed(ti.ui.LEFT, 'a'):
                goal[0][0] -= .05 if goal[0][0] > 0.08 else 0
            if window.is_pressed(ti.ui.RIGHT, 'd'):
                goal[0][0] += .05 if goal[0][0] < 0.91 else 0
            if window.is_pressed(ti.ui.UP, 'w'):
                goal[0][1] += .1 if goal[0][1] < 0.85 else 0
            if window.is_pressed(ti.ui.DOWN, 's'):
                goal[0][1] -= .1 if goal[0][1] > 0.15 else 0
            if window.is_pressed('v'):
                verbose = not verbose
            if window.is_pressed(ti.ui.CTRL):
                time_integrate_method = "Forward Euler" if time_integrate_method == "RK-2" else "RK-2"
                print(f"Change to {time_integrate_method} time integration.")
            i = 0

    for _ in range(sub_step_num):
        ret = substep(time_integrate_method == "RK-2")

    # Draw the robot body.
    transform.ApplyToPoints(material_space_robot_body_vertex, robot_body_vertex)
    canvas.lines(robot_body_vertex, 0.04, color=(0.1, 0.7, 0.3))

    # Draw the propellers.
    transform.ApplyToPoints(material_space_robot_propeller_vertex, robot_propeller_vertex)
    canvas.lines(robot_propeller_vertex, 0.01, robot_propeller_index, (1., 0.7, 0.2))
    
    if verbose:
        # Draw the target point.
        canvas.circles(goal, 0.01, color=(1., 0.1, 0.1))

        # Visualize the thrust.
        VisualizeForce(ret[0], ret[1])
        canvas.lines(force_visualization_vertex, 0.006, force_visualization_index, (0.3, 0.3, 1))
    
    window.show()
    i += 1
