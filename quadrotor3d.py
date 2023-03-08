from core import *
import trimesh
import numpy as np

# Load mesh.
body_mesh = []
for i in range(1, 6):
    mesh = trimesh.load_mesh(f"./body{i}.obj")
    v = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0 = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0.from_numpy(np.array(mesh.vertices[:, (1, 2, 0)] * 0.5, dtype=np.float32))
    t = ti.field(int, shape=3 * len(mesh.faces))
    t.from_numpy(np.array(mesh.faces, dtype=np.int32).ravel())
    body_mesh.append((v0, v, t))

propeller_mesh = []
for i in range(1, 5):
    mesh = trimesh.load_mesh(f"./propeller{i}.obj")
    v = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0 = ti.Vector.field(3, float, shape=(len(mesh.vertices),))
    v0.from_numpy(np.array(mesh.vertices[:, (1, 2, 0)] * 0.5, dtype=np.float32))
    t = ti.field(int, shape=3 * len(mesh.faces))
    t.from_numpy(np.array(mesh.faces, dtype=np.int32).ravel())
    propeller_mesh.append((v0, v, t))

# Relative location of propellers.
rel_loc = ti.Vector.field(3, float, shape=(4, ))
rel_loc[0] = [0, 0, 0.25]
rel_loc[1] = [0.25, 0, 0]
rel_loc[2] = [0, 0, -0.25]
rel_loc[3] = [-0.25, 0, 0]

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

@ti.kernel
def Initialize():
    transform.translation[None] = [0., 1.8, 0.]
    # Note t
    transform.rotation[None] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    goal[0] = [0., 1.8, 0.]

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
    var_q_dot["angular_velocity"] = Jinv @ (external_torque[None] - w.cross_product(J @ w))
    return var_q_dot

@ti.kernel
def ForwardEuler():
    var_q = {"translation": transform.translation[None], "rotmat": transform.rotation[None], "rotvec": tm.vec3([0, 0, 0]), 
             "velocity": velocity[None], "angular_velocity": angular_velocity[None]}
    var_q_dot = F(var_q)
    transform.translation[None] += var_q_dot["translation"] * time_step
    transform.StepRotationBythrustRotationVector(var_q_dot["rotvec"] * time_step)
    velocity[None] += var_q_dot["velocity"] * time_step
    angular_velocity[None] += var_q_dot["angular_velocity"] * time_step

@ti.kernel
def RungeKutta2():
    pass

@ti.kernel
def ApplyForce(thrust1: float, thrust2: float, thrust3: float, thrust4: float):
    external_force[None] = gravitational_acceleration * mass
    g = external_force[None].norm()
    force_direction = transform.rotation[None] @ tm.vec3([0, 0, 1])
    external_force[None] += force_direction * (thrust1 + thrust2 + thrust3 + thrust4 + 1) * g
    external_torque[None] = [0, 0, 0]
    for i in rel_loc:
        external_torque[None] += rel_loc[i].cross_product(force_direction)
    external_torque[None] += (thrust1 - thrust2 + thrust3 - thrust4) * force_direction * 0.01

# PD Control.
@ti.kernel
def YControllor(y_goal: float) -> float:
    # PD control for y.
    target = clamp(y_goal, 0, 1)
    y = transform.translation[None][1]
    dot_y = velocity[None][1]
    return (- 15 * (y - target) - 15 * dot_y + 0.5) * gravity

@ti.kernel
def ThetaControllor(theta_goal: float) -> float:
    # PD control for theta.
    target = clamp(theta_goal, -tm.pi / 30, tm.pi / 30)
    return (9 * (theta[None] - target) + 15 * angular_velocity[None]) * gravity

@ti.kernel
def PhiControllor(phi_goal: float) -> float:
    pass

@ti.kernel
def PsiControllor(psi_goal: float) -> float:
    pass

@ti.kernel
def XControllor(x_goal: float) -> float:
    # X controllor is a 2nd-level controllor.
    target = clamp(x_goal, -0.5, 0.5)
    x = transform.translation[None][0]
    dot_x = velocity[None][0]
    return 5 * (x - target) + 15 * dot_x - theta[None]

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
        
    canvas.scene(scene)
    window.show()
