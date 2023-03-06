import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu)

@ti.data_oriented
class Transform2D:
    def __init__(self) -> None:
        self.rotation = ti.Matrix.field(2, 2, float, shape=())
        self.translation = ti.Vector.field(2, float, shape=())
    
    @ti.func
    def ApplyToPoint(self, pt):
        return self.rotation[None] @ pt + self.translation[None]
    
    @ti.kernel
    def ApplyToPoints(self, pts: ti.template(), ret: ti.template()):
        for i in pts:
            ret[i] = self.ApplyToPoint(pts[i])

    @ti.func
    def UpdateRotationFromTheta(self, t):   
        self.rotation[None] = [[tm.cos(t), -tm.sin(t)], [tm.sin(t), tm.cos(t)]]
    
    @ti.func
    def Inverse(self):
        ret_rotation = self.rotation.transpose()
        ret_translation = - ret_rotation @ self.translation
        return Transform2D(ret_rotation, ret_translation)
    
    
@ti.data_oriented
class Transform3D:
    def __init__(self) -> None:
        self.rotation = ti.Matrix.field(3, 3, float, shape=())
        self.translation = ti.Vector.field(3, float, shape=())
    
    @ti.func
    def ApplyToPoint(self, pt):
        return self.rotation[None] @ pt + self.translation[None]
    
    @ti.kernel
    def ApplyToPoints(self, pts: ti.template(), ret: ti.template()):
        for i in pts:
            ret[i] = self.ApplyToPoint(pts[i])

    @ti.func
    def StepRotationByAngleAxis(self, axis_angle):
        t = axis_angle.norm()
        axis = axis_angle / t
        c = tm.cos(t)
        s = tm.sin(t)
        backup_rotation = self.rotation[None]
        self.rotation[None] = [[c, 0, 0], [0, c, 0], [0, 0, c]]
        self.rotation[None] += (1 - c) * axis.outer_product(axis) + s * ToSkewSymmetric3D(axis)
        self.rotation[None] = backup_rotation @ self.rotation[None]
    
    @ti.func
    def Inverse(self):
        ret_rotation = self.rotation.transpose()
        ret_translation = - ret_rotation @ self.translation
        return Transform3D(ret_rotation, ret_translation)

@ti.func
def ToSkewSymmetric2D(w):
    return tm.mat2([[0, -w[None]], [w[None], 0]])

@ti.func
def FromSkewSymmetric2D(m):
    return m[1, 0]

@ti.func
def ToSkewSymmetric3D(w):
    return tm.mat3([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

@ti.func
def FromSkewSymmetric3D(m):
    return tm.vec3([m[2, 1], m[0, 2], m[1, 0]])

@ti.func
def clamp(v, low, high):
    ret = v
    if v < low:
        ret = low
    if v > high:
        ret = high
    return ret