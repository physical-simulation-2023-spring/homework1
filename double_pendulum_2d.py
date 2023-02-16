import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu)

num_links = 2

position = ti.Vector.field(2, dtype=float, shape=(num_links, ))
velocity = ti.Vector.field(2, dtype=float, shape=(num_links, ))
angular_velocity = ti.Vector.field(1, dtype=float, shape=(num_links, ))
rotation_matrices = ti.Matrix.field(2, 2, dtype=float, shape=(num_links, ))
generalized_position = ti.Vector([tm.pi / 4, 0.])
generalized_velocity = ti.Vector([0., 0.])



print(position[0], rotation_matrices[0])
print(rotation_matrices[0] @ position[0])
