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

# The velocity and the angular velocity of the object.
velocity = tm.vec2()
angular_velocity = ti.float32()

external_force = tm.vec2()
external_torque = ti.float32()

time_step = 0.001
gravity = tm.vec2([0, -2])
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
    pass

@ti.func
def RK2():
    pass

@ti.func
def RK4():
    pass

@ti.kernel
def step():
    pass

@ti.func
def test_ret_of_pts(pts, ret):
    for i in pts:
        pts[i] = [1, 0]

@ti.kernel
def test_main():
    test_ret_of_pts(robot_wings_coord, robot_wings_index)


print(robot_wings_coord)


test_main()
print(robot_wings_coord)
# window = ti.ui.Window('Window Title', res = (640, 360), pos = (150, 150))

# while window.running:
#     step()
#     canvas = window.get_canvas()
#     canvas.set_background_color((0.1, 0.2, 0.8))
    
#     # Draw the robot body.
#     line_field = ti.Vector.field(2, dtype=float, shape=(num_links + 1, ))
#     # Draw the robot wings.
#     canvas.circles(position, ball_radius, (1., 0.7, 0.2))