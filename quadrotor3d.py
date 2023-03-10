from release.util import *
import trimesh
import numpy as np

# Load mesh.
body_m = np.array([[np.sqrt(2) / 4, 0, np.sqrt(2) / 4], [0, .5, 0], [-np.sqrt(2) / 4, 0, np.sqrt(2) / 4]])
body_mesh = []
for i in range(1, 6):
    mesh = trimesh.load_mesh(f"./release/mesh/body{i}.obj")
    v = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0 = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0.from_numpy(np.array(mesh.vertices[:, (1, 2, 0)] @ body_m, dtype=np.float32))
    t = ti.field(int, shape=3 * len(mesh.faces))
    t.from_numpy(np.array(mesh.faces, dtype=np.int32).ravel())
    body_mesh.append((v0, v, t))
propeller_mesh = []
for i in range(1, 5):
    mesh = trimesh.load_mesh(f"./release/mesh/propeller{i}.obj")
    v = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0 = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0.from_numpy(np.array(mesh.vertices[:, (1, 2, 0)] @ body_m, dtype=np.float32))
    t = ti.field(int, shape=3 * len(mesh.faces))
    t.from_numpy(np.array(mesh.faces, dtype=np.int32).ravel())
    propeller_mesh.append((v0, v, t))

# Relative location of propellers.
rel_loc = ti.Vector.field(3, float, shape=(4, ))
FillData(rel_loc, [[0.2, 0, .2], [0.2, 0, -0.2], [-0.2, 0, -0.2], [-0.2, 0, 0.2]])

# The global transformation of the object.
transform = Transform3D()

# The velocity and the angular velocity of the object.
velocity = ti.Vector.field(3, float, shape=())
angular_velocity = ti.Vector.field(3, float, shape=())

external_force = ti.Vector.field(3, float, shape=())
external_torque = ti.Vector.field(3, float, shape=())

fps = 60
time_step = 0.001
sub_step_num = int(1 / fps / time_step)

mass = 4.
gravitational_acceleration = tm.vec3([0, -9.8, 0])
gravity = (gravitational_acceleration * mass).norm()
body_inertia = tm.mat3([[.1, 0, 0], [0, .05, 0], [0, 0, .1]])
body_inertia_inv = tm.mat3([[10., 0, 0], [0, 20., 0], [0, 0, 10.]])

goal = ti.Vector.field(3, float, shape=(1, ))

force_visualization_vertex = ti.Vector.field(2, float, shape=(8, ))
force_visualization_index = ti.Vector.field(2, int, shape=(8, ))
FillData(force_visualization_vertex, sum([[[0.6 + i * 0.1, 0.1], [0.6 + i * 0.1, 0.2]] for i in range(4)], []))
FillData(force_visualization_index, [[i, i + 1] for i in range(0, 15, 2)])

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
def F(q):
    # Newton-Euler equation: compute time derivative of q.
    J = ComputeInertia(q["rotmat"])
    Jinv = ComputeInertiaInv(q["rotmat"])
    q_dot = {var: ti.zero(q[var]) for var in q}
    q_dot["translation"] = q["velocity"]
    q_dot["rotvec"] = q["angular_velocity"]
    q_dot["velocity"] = external_force[None] / mass
    w = q["angular_velocity"]
    q_dot["angular_velocity"] = Jinv @ (external_torque[None] - w.cross(J @ w))
    return q_dot

@ti.func
def ForwardEuler():
    q = {"translation": transform.translation[None], "rotmat": transform.rotation[None], "rotvec": tm.vec3([0, 0, 0]), 
             "velocity": velocity[None], "angular_velocity": angular_velocity[None]}
    q_dot = F(q)
    transform.translation[None] += q_dot["translation"] * time_step
    transform.rotation[None] = StepRotationByDeltaRotationVector(transform.rotation[None], q_dot["rotvec"], time_step)
    velocity[None] += q_dot["velocity"] * time_step
    angular_velocity[None] += q_dot["angular_velocity"] * time_step

@ti.func
def RungeKutta2():
    q = {"translation": transform.translation[None], "rotmat": transform.rotation[None], "rotvec": tm.vec3([0, 0, 0]), 
             "velocity": velocity[None], "angular_velocity": angular_velocity[None]}
    q_dot_half = F(q)
    h_half = time_step / 2
    q_half = {
        "translation": q["translation"] + q_dot_half["translation"] * h_half, 
        "rotmat": StepRotationByDeltaRotationVector(transform.rotation[None], q_dot_half["rotvec"], h_half),
        "rotvec": tm.vec3([0, 0, 0]),
        "velocity": q["velocity"] + q_dot_half["velocity"] * h_half, 
        "angular_velocity": q["angular_velocity"] + q_dot_half["angular_velocity"] * h_half
    }
    q_dot = F(q_half)
    transform.translation[None] += q_dot["translation"] * time_step
    transform.rotation[None] = StepRotationByDeltaRotationVector(transform.rotation[None], q_dot["rotvec"], time_step)
    velocity[None] += q_dot["velocity"] * time_step
    angular_velocity[None] += q_dot["angular_velocity"] * time_step 

@ti.func
def ApplyForce(thrust_input):
    thrust = thrust_input
    for i in range(4):
        thrust[i] = clamp(thrust_input[i], 0, 19.6)
    external_force[None] = gravitational_acceleration * mass
    force_direction = transform.rotation[None] @ tm.vec3([0, 1, 0])
    external_force[None] += force_direction * thrust.sum()
    external_torque[None] = [0, 0, 0]
    for i in rel_loc:
        rotate_rel_loc = transform.rotation[None] @ rel_loc[i]
        external_torque[None] += rotate_rel_loc.cross(force_direction * thrust[i])
    # Rotating propellers generate additional torque.
    # Let opposite propellers rotate in the same direction.
    external_torque[None] += (thrust[0] - thrust[1] + thrust[2] - thrust[3]) * force_direction * 0.1

# PD Control.
@ti.func
def YControllor(y_goal: float) -> float:
    # PD control for y.
    target = clamp(y_goal, 0, 2)
    y = transform.translation[None][1]
    dot_y = velocity[None][1]
    return (- 16 * (y - target) - 10 * dot_y + 0.25) * gravity

@ti.func
def OrientationControllor(orientation_goal: tm.vec2) -> tm.vec4:
    # PD control for angle.
    ypr = transform.GetYawPitchRoll()
    yaw, pitch, roll = ypr[0], ypr[1], ypr[2]
    target_pitch = clamp(orientation_goal[0], -tm.pi / 15, tm.pi / 15)
    target_roll = clamp(orientation_goal[1], -tm.pi / 15, tm.pi / 15)
    delta_pitch = (40 * (pitch - target_pitch) - 6 * angular_velocity[None][0]) * gravity
    delta_roll = (40 * (roll - target_roll) - 6 * angular_velocity[None][2]) * gravity
    delta_yaw = (- 0.5 * yaw + 0.15 * angular_velocity[None][1]) * gravity
    ret1 = - delta_pitch + delta_roll - delta_yaw
    ret2 = + delta_pitch + delta_roll + delta_yaw
    ret3 = + delta_pitch - delta_roll - delta_yaw
    ret4 = - delta_pitch - delta_roll + delta_yaw
    return tm.vec4([ret1, ret2, ret3, ret4])

@ti.func
def XZControllor(x_goal: float, z_goal: float) -> tm.vec2:
    # XZ controllor is a 2nd-level controllor.
    x_target, z_target = clamp(x_goal, -1, 1), clamp(z_goal, -1, 1)
    x, z = transform.translation[None][0], transform.translation[None][2]
    dot_x, dot_z = velocity[None][0], velocity[None][2]
    pitch_goal, roll_goal = - (z - z_target) - dot_z, (x - x_target) + dot_x
    return tm.vec2([pitch_goal, roll_goal])

@ti.kernel
def substep(use_RK2: bool) -> tm.vec4:
    main_thrust = YControllor(goal[0][1])
    ret = OrientationControllor(XZControllor(goal[0][0], goal[0][2])) + main_thrust
    ApplyForce(ret)
    if use_RK2:
        RungeKutta2()
    else:
        ForwardEuler()
    return ret

@ti.kernel
def VisualizeForce(thrust: tm.vec4):
    for i in range(4):
        force_visualization_vertex[i * 2 + 1][1] = clamp(thrust[i], 0, 19.6) / 100 + 0.1

i = 0
time_integrate_method = "Forward Euler"
verbose = False

window = ti.ui.Window("Quadrotor 3D", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

Initialize()

video_manager = ti.tools.VideoManager(output_dir="./output3d", framerate=60, automatic_build=False)

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
            if window.is_pressed('v'):
                verbose = not verbose
            if window.is_pressed(ti.ui.CTRL):
                time_integrate_method = "Forward Euler" if time_integrate_method == "RK-2" else "RK-2"
                print(f"Change to {time_integrate_method} time integration.")
            i = 0

    for _ in range(sub_step_num):
        ret = substep(time_integrate_method == "RK-2")

    camera.position(0.0, 2, -2.0)
    camera.lookat(0.0, 1., 0.0)
    scene.set_camera(camera)
    
    scene.point_light(pos=(0, 2, 0), color=(0.8, 0.8, 0.8))
    scene.ambient_light((0.5, 0.5, 0.5))
    
    # Draw the robot body.
    for mesh in body_mesh:
        transform.ApplyToPoints(mesh[0], mesh[1])
        scene.mesh(mesh[1], mesh[2], color=(1., 0.7, 0.3))

    # Draw the propellers.
    for s in [-2, 2]:
        for mesh in propeller_mesh[::s]:
            transform.ApplyToPoints(mesh[0], mesh[1])
            scene.mesh(mesh[1], mesh[2], color=(0.6 + s * 0.2, 0.2, 0.6 - s * 0.2))

    if verbose:
        # Draw the target point.
        scene.particles(goal, 0.02, color=(0.1, 0.8, 0.1))
        
        # Visualize the thrust.
        VisualizeForce(ret)
        canvas.lines(force_visualization_vertex, 0.03, force_visualization_index, color=(0.1, 0.7, 0.3))

    canvas.scene(scene)
    video_manager.write_frame(window.get_image_buffer_as_numpy())
    window.show()
    i += 1

video_manager.make_video(gif=True, mp4=False)
