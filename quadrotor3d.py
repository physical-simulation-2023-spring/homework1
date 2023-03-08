from core import *
import trimesh
import numpy as np

# Load mesh.
body_m = np.array([[np.sqrt(2) / 4, 0, np.sqrt(2) / 4], [0, .5, 0], [-np.sqrt(2) / 4, 0, np.sqrt(2) / 4]])
body_mesh = []
for i in range(1, 6):
    mesh = trimesh.load_mesh(f"./body{i}.obj")
    v = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0 = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0.from_numpy(np.array(mesh.vertices[:, (1, 2, 0)] @ body_m, dtype=np.float32))
    t = ti.field(int, shape=3 * len(mesh.faces))
    t.from_numpy(np.array(mesh.faces, dtype=np.int32).ravel())
    body_mesh.append((v0, v, t))
propeller_mesh = []
for i in range(1, 5):
    mesh = trimesh.load_mesh(f"./propeller{i}.obj")
    v = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0 = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0.from_numpy(np.array(mesh.vertices[:, (1, 2, 0)] @ body_m, dtype=np.float32))
    t = ti.field(int, shape=3 * len(mesh.faces))
    t.from_numpy(np.array(mesh.faces, dtype=np.int32).ravel())
    propeller_mesh.append((v0, v, t))

# Relative location of propellers.
# 1   4
#  \ /
#  / \
# /   \ 
#2     3
rel_loc = ti.Vector.field(3, float, shape=(4, ))
FillData(rel_loc, [[0.25, 0, 0.25], [0.25, 0, -0.25], [-0.25, 0, -0.25], [-0.25, 0, 0.25]])

# The global transformation of the object.
transform = Transform3D()

# The velocity and the angular velocity of the object.
velocity = ti.Vector.field(3, float, shape=())
angular_velocity = ti.Vector.field(3, float, shape=())

var_name = ["translation", "rotvec", "velocity", "angular_velocity"]

external_force = ti.Vector.field(3, float, shape=())
external_torque = ti.Vector.field(3, float, shape=())

fps = 60
time_step = 0.001
sub_step_num = int(1 / fps / time_step)

mass = 4.
gravitational_acceleration = tm.vec2([0, -9.8])
gravity = (gravitational_acceleration * mass).norm()
body_inertia = tm.mat3([[2., 0, 0], [0, 2., 0], [0, 0, 4.]])
body_inertia_inv = tm.mat3([[.5, 0, 0], [0, .5, 0], [0, 0, .25]])

goal = ti.Vector.field(3, float, shape=(1, ))

force_vis_temp = ti.Vector.field(3, float, shape=(4, ))

@ti.kernel
def Initialize():
    transform.translation[None] = [0., 1, 0.]
    transform.rotation[None] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    velocity[None] = [0, 0, 0]
    angular_velocity[None] = [0, 0, 0]
    goal[0] = [0., 1, 0.]

@ti.func
def ComputeInertia(rotmat):
    return rotmat @ body_inertia @ rotmat.transpose()

@ti.func
def ComputeInertiaInv(rotmat):
    return rotmat @ body_inertia_inv @ rotmat.transpose()

@ti.func
def F(var_q):
    # Newton-Euler equation: compute time derivative of var_q.
    J = ComputeInertia(var_q["rotmat"])
    Jinv = ComputeInertiaInv(var_q["rotmat"])
    var_q_dot = {var: ti.zero(var_q[var]) for var in var_name}
    var_q_dot["translation"] = var_q["velocity"]
    var_q_dot["rotvec"] = var_q["angular_velocity"]
    var_q_dot["velocity"] = external_force[None] / mass
    w = var_q["angular_velocity"][None]
    var_q_dot["angular_velocity"] = Jinv @ (external_torque[None] - w.cross(J @ w))
    return var_q_dot

@ti.kernel
def ForwardEuler():
    var_q = {"translation": transform.translation[None], "rotmat": transform.rotation[None], "rotvec": tm.vec3([0, 0, 0]), 
             "velocity": velocity[None], "angular_velocity": angular_velocity[None]}
    var_q_dot = F(var_q)
    transform.translation[None] += var_q_dot["translation"] * time_step
    transform.rotation[None] = StepRotationByDeltaRotationVector(transform.rotation[None], var_q_dot["rotvec"] * time_step)
    velocity[None] += var_q_dot["velocity"] * time_step
    angular_velocity[None] += var_q_dot["angular_velocity"] * time_step

@ti.kernel
def RungeKutta2():
    var_q = {"translation": transform.translation[None], "rotmat": transform.rotation[None], "rotvec": tm.vec3([0, 0, 0]), 
             "velocity": velocity[None], "angular_velocity": angular_velocity[None]}
    var_q_dot_half = F(var_q)
    h_half = time_step / 2
    var_q_half = {
        "translation": var_q["translation"] + var_q_dot_half["translation"] * h_half, 
        "rotmat": StepRotationByDeltaRotationVector(transform.rotation[None], var_q_dot_half["rotvec"] * h_half),  
        "velocity": var_q["velocity"] + var_q_dot_half["velocity"] * h_half, 
        "angular_velocity": var_q["angular_velocity"] + var_q_dot_half["angular_velocity"] * h_half
    }
    var_q_dot = F(var_q_half)
    transform.translation[None] += var_q_dot["translation"] * time_step
    transform.rotation[None] = StepRotationByDeltaRotationVector(transform.rotation[None], var_q_dot["rotvec"] * time_step)
    velocity[None] += var_q_dot["velocity"] * time_step
    angular_velocity[None] += var_q_dot["angular_velocity"] * time_step 

@ti.func
def ApplyForce(thrust1: float, thrust2: float, thrust3: float, thrust4: float):
    external_force[None] = gravitational_acceleration * mass
    g = external_force[None].norm()
    force_direction = transform.rotation[None] @ tm.vec3([0, 1, 0])
    external_force[None] += force_direction * (thrust1 + thrust2 + thrust3 + thrust4) * g
    external_torque[None] = [0, 0, 0]
    for i in rel_loc:
        rotate_rel_loc = transform.rotation[None] @ rel_loc[i]
        external_torque[None] += rotate_rel_loc.cross(force_direction)
    external_torque[None] += (thrust1 - thrust2 + thrust3 - thrust4) * force_direction * 0.01

# PD Control.
@ti.func
def YControllor(y_goal: float) -> float:
    # PD control for y.
    target = clamp(y_goal, 0, 2)
    y = transform.translation[None][1]
    dot_y = velocity[None][1]
    return (- 15 * (y - target) - 15 * dot_y + 0.25) * gravity

@ti.func
def OrientationControllor(orientation_goal):
    # PD control for angle.
    # Assume (1, 2) for pitch, (3, 4) for roll right now.
    ypr = transform.GetYawPitchRoll()
    yaw, pitch, roll = ypr[0], ypr[1], ypr[2]
    target_pitch = clamp(orientation_goal[0], -tm.pi / 15, tm.pi / 15)
    target_roll = clamp(orientation_goal[1], -tm.pi / 15, tm.pi / 15)
    ret1 = (9 * (pitch - target_pitch) + 15 * angular_velocity[None][0]) * gravity
    ret2 = ret1
    ret3 = - ret1
    ret4 = ret3
    delta_roll = (9 * (roll - target_roll) + 15 * angular_velocity[None][1]) * gravity
    ret1 += delta_roll
    ret2 -= delta_roll
    ret3 -= delta_roll
    ret4 += delta_roll
    delta_yaw = (9 * yaw + 15 * angular_velocity[None][2]) * gravity
    ret1 += delta_yaw
    ret3 += delta_yaw
    ret2 -= delta_yaw
    ret4 -= delta_yaw
    return tm.vec4([ret1, ret2, ret3, ret4])

@ti.func
def XZControllor(x_goal, z_goal):
    # XZ controllor is a 2nd-level controllor.
    ypr = transform.GetYawPitchRoll()
    pitch, roll = ypr[1], ypr[2]
    x_target = clamp(x_goal, -1, 1)
    z_target = clamp(z_goal, -1, 1)
    x = transform.translation[None][0]
    z = transform.translation[None][2]
    dot_x = velocity[None][0]
    dot_z = velocity[None][2]
    pitch_goal = 5 * (x - x_target) + 15 * dot_x - pitch
    roll_goal = 5 * (z - z_target) + 15 * dot_z - roll
    return tm.vec2([pitch_goal, roll_goal])

@ti.kernel
def substep(use_RK2: bool) -> tm.vec4:
    main_thrust = YControllor(goal[0][1])
    thrust_delta = OrientationControllor(XZControllor(goal[0][0], goal[0][2]))
    ApplyForce(main_thrust + thrust_delta[0], main_thrust + thrust_delta[1], main_thrust + thrust_delta[2], main_thrust + thrust_delta[3])
    # if use_RK2:
    #     RungeKutta2()
    # else:
    #     ForwardEuler()
    return tm.vec4([main_thrust + thrust_delta[0], main_thrust + thrust_delta[1], main_thrust + thrust_delta[2], main_thrust + thrust_delta[3]])
        
@ti.kernel
def plot_force_temp(forces: tm.vec4):
    force_direction = transform.rotation[None] @ tm.vec3([0, 1, 0])
    for i in force_vis_temp:
        force_vis_temp[i] = transform.ApplyToPoint(rel_loc[i]) + force_direction * forces[i] * 0.03

i = 0
time_integrate_method = ["Forward Euler", "RK-2", "RK-4"]
time_integrate_method_index = 0

window = ti.ui.Window("Quadrotor 3D", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

Initialize()
current_t = 0.0

while window.running:
    if window.get_event(ti.ui.PRESS):
        if i > 10:
            if window.event.key == 'r': 
                Initialize()
            elif window.event.key in [ti.ui.ESCAPE]: 
                break
            if window.is_pressed(ti.ui.LEFT, 'a'):
                goal[0][0] += .05 if goal[0][0] > -0.91 else 0
            if window.is_pressed(ti.ui.RIGHT, 'd'):
                goal[0][0] -= .05 if goal[0][0] < 0.91 else 0
            if window.is_pressed(ti.ui.UP):
                goal[0][1] += .1 if goal[0][1] < 1.85 else 0
            if window.is_pressed(ti.ui.DOWN):
                goal[0][1] -= .1 if goal[0][1] > 0.15 else 0
            if window.is_pressed('w'):
                goal[0][2] += .05 if goal[0][2] < 0.91 else 0
            if window.is_pressed('s'):
                goal[0][2] -= .05 if goal[0][2] > -0.91 else 0
            if window.is_pressed(ti.ui.CTRL):
                time_integrate_method = "Forward Euler" if time_integrate_method == "RK-2" else "RK-2"
                print(f"Change to {time_integrate_method} time integration.")
            i = 0

    camera.position(0.0, 0.5, -2.0)
    camera.lookat(0.0, 1., 0.0)
    scene.set_camera(camera)
    
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    
    for mesh in body_mesh:
        transform.ApplyToPoints(mesh[0], mesh[1])
        scene.mesh(mesh[1], mesh[2], color=(1., 0.7, 0.2))

    for mesh in propeller_mesh[::2]:
        transform.ApplyToPoints(mesh[0], mesh[1])
        scene.mesh(mesh[1], mesh[2], color=(1., 0.2, 0.2))
        
    for mesh in propeller_mesh[::-2]:
        transform.ApplyToPoints(mesh[0], mesh[1])
        scene.mesh(mesh[1], mesh[2], color=(0.1, 0.1, 0.8))
    
    # ret = substep(False)
    scene.particles(goal, 0.02, color=(1., 0.1, 0.1))
    plot_force_temp(tm.vec4([-20, -10, 10, 20]))
    scene.particles(force_vis_temp, 0.04, color=(0.2, 0.2, 1))
    canvas.scene(scene)
    window.show()

    i += 1
