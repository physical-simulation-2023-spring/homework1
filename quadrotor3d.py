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
trans = Transform3D()

# The velocity and the angular velocity of the object.
velocity = ti.Vector.field(3, float, shape=())
angular_velocity = ti.Vector.field(3, float, shape=())

external_force = ti.Vector.field(3, float, shape=())
external_torque = ti.Vector.field(3, float, shape=())

fps = 60
time_step = 0.001
sub_step_num = int(1 / fps / time_step)

mass = 4.
gravitational_acceleration = tm.vec2([0, -9.8])
body_inertia = tm.mat3([[2., 0, 0], [0, 2., 0], [0, 0, 4.]])
body_inertia_inv = tm.mat3([[.5, 0, 0], [0, .5, 0], [0, 0, .25]])

goal = ti.Vector.field(3, float, shape=(1, ))

@ti.kernel
def Initialize():
    trans.translation[None] = [0., 1.8, 0.]
    trans.rotation[None] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    goal[0] = [0., 1.8, 0.]

@ti.func
def ComputeInertia():
    return trans.rotation[None] @ body_inertia @ trans.rotation[None]

@ti.func
def ComputeInertiaInv():
    return trans.rotation[None] @ body_inertia_inv @ trans.rotation[None]

@ti.kernel
def ForwardEuler():
    trans.translation[None] += velocity[None] * time_step
    trans.StepRotationByAngleAxis(angular_velocity[None] * time_step)
    velocity[None] += external_force[None] * time_step / mass
    J = ComputeInertia()
    Jinv = ComputeInertiaInv()
    w = angular_velocity[None]
    angular_velocity[None] += Jinv @ (external_torque[None] - w.cross_product(J @ w)) * time_step
    
@ti.kernel
def ApplyForce(delta1: float, delta2: float, delta3: float, delta4: float):
    external_force[None] = gravitational_acceleration * mass
    g = external_force[None].norm()
    force_direction = tm.vec2([trans.rotation[None][1, 0], trans.rotation[None][1, 1]])
    external_force[None] += force_direction * (delta1 + delta2 + delta3 + delta4 + 1) * g
    external_torque[None] = (left_delta - right_delta) * g * 0.25

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
# initialize_mass_points()

while window.running:
    # if current_t > 1.5:
    #     # Reset
    #     initialize_mass_points()
    #     current_t = 0

    # for i in range(substeps):
    #     substep()
    #     current_t += dt
    # update_vertices()

    camera.position(0.0, 0.5, -2.0)
    camera.lookat(0.0, 1., 0.0)
    scene.set_camera(camera)
    
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    
    for mesh in body_mesh:
        trans.ApplyToPoints(mesh[0], mesh[1])
        scene.mesh(mesh[1], mesh[2], color=(1., 0.7, 0.2))

    for mesh in propeller_mesh[::2]:
        trans.ApplyToPoints(mesh[0], mesh[1])
        scene.mesh(mesh[1], mesh[2], color=(1., 0.2, 0.2))
        
    for mesh in propeller_mesh[::-2]:
        trans.ApplyToPoints(mesh[0], mesh[1])
        scene.mesh(mesh[1], mesh[2], color=(0.1, 0.1, 0.8))
        
    canvas.scene(scene)
    window.show()