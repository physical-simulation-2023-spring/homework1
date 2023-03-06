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

external_force = ti.Vector.field(2, float, shape=())
external_torque = ti.field(float, shape=())

fps = 60
time_step = 0.001
sub_step_num = int(1 / fps / time_step)

mass = 4.
gravitational_acceleration = tm.vec2([0, -9.8])
body_inertia = 0.5

goal = ti.Vector.field(2, float, shape=(1, ))

# You may want to define your own variables or functions here.
# -- YOUR CODE BEGINS --


# -- END OF YOUR CODE --

@ti.kernel
def Initialize():
    trans.translation[None] = [0, 0.5]
    trans.rotation[None] = [[1., 0.], [0., 1.]]
    theta[None] = 0
    goal[0] = [0.5, 0.5]
    # You may want to initialize your variables here.
    # -- YOUR CODE BEGINS --


    # -- END OF YOUR CODE --

@ti.kernel
def ForwardEuler():
    trans.translation[None] += velocity[None] * time_step
    theta[None] += angular_velocity[None] * time_step
    trans.UpdateRotationFromTheta(theta[None])
    velocity[None] += external_force[None] * time_step / mass
    angular_velocity[None] += external_torque[None] * time_step / body_inertia

@ti.kernel
def ApplyForce(left_delta: float, right_delta: float):
    external_force[None] = gravitational_acceleration * mass
    g = external_force[None].norm()
    force_direction = tm.vec2([trans.rotation[None][1, 0], trans.rotation[None][1, 1]])
    external_force[None] += force_direction * (left_delta + right_delta + 1) * g
    external_torque[None] = (left_delta - right_delta) * g * 0.25

@ti.kernel
def RungeKutta2():
    # -- YOUR CODE BEGINS --
    pass
    # -- END OF YOUR CODE --
    

@ti.kernel
def RungeKutta4():
    # -- YOUR CODE BEGINS --
    pass
    # -- END OF YOUR CODE --

@ti.kernel
def YControllor(y_goal: float) -> float:
    # PD control for y.
    target = clamp(y_goal, 0.1, 0.9)
    y = trans.translation[None][1]
    dot_y = velocity[None][1]
    return - 6. * (y - target) - 15. * dot_y

@ti.kernel
def ThetaControllor(theta_goal: float) -> float:
    # PD control for theta.
    target = clamp(theta_goal, -tm.pi / 30, tm.pi / 30)
    return - 30. * (theta[None] - target) - 10. * angular_velocity[None]

@ti.kernel
def XControllor(x_goal: float) -> float:
    # X controllor is a 2nd level controllor.
    target = clamp(x_goal, -0.4, 0.4)
    x = trans.translation[None][0]
    dot_x = velocity[None][0]
    return - (x - target) - 2 * dot_x

i = 0
time_integrate_method = ["Forward Euler", "RK-2", "RK-4"]
time_integrate_method_index = 0

window = ti.ui.Window('Quadrotor 2D', res = (640, 360), pos = (150, 150), vsync=True)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((0.25, 0.25, 0.25))

Initialize()

while window.running:
    if window.get_event(ti.ui.PRESS):
        if i > 20:
            if window.event.key == 'r': 
                Initialize()
            elif window.event.key in [ti.ui.ESCAPE]: 
                break
            if window.is_pressed(ti.ui.LEFT, 'a'):
                goal[0][0] -= .05
            if window.is_pressed(ti.ui.RIGHT, 'd'):
                goal[0][0] += .05
            if window.is_pressed(ti.ui.UP, 'w'):
                goal[0][1] += .1
            if window.is_pressed(ti.ui.DOWN, 's'):
                goal[0][1] -= .1
            if window.is_pressed(ti.ui.SHIFT):
                time_integrate_method_index = (time_integrate_method_index + 1) % 3
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
        if time_integrate_method_index == 0:
            ForwardEuler()
        elif time_integrate_method_index == 1:
            RungeKutta2()
        else:
            RungeKutta4()
        
    i += 1

    window.show()
