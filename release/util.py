import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu)

@ti.data_oriented
class Transform2D:
    def __init__(self):
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
        self.rotation[None] = tm.rotation2d(t)


@ti.data_oriented
class Transform3D:
    def __init__(self):
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
    def GetYawPitchRoll(self):
        R = self.rotation[None]
        t = tm.sqrt(R[0,1] * R[0,1] + R[1,1] * R[1,1])
        # Small t makes error. It's OK if we do not have large pitch.
        x = tm.atan2(-R[2,0], R[2,2])
        y = tm.atan2( R[2,1], t)
        z = tm.atan2(-R[0,1], R[1,1])
        return tm.vec3([x, y, z])


@ti.func
def StepRotationByDeltaRotationVector(rotmat, w, h):
    t = w.norm()
    ret_mat = rotmat
    if t > 1e-7:
        axis = w / t
        m = tm.rot_by_axis(axis, t * h)
        ret_mat = m[:3, :3] @ rotmat
    return ret_mat

@ti.func
def clamp(v, low, high):
    ret = v
    if v < low:
        ret = low
    if v > high:
        ret = high
    return ret

def FillData(field, data):
    for i, d in enumerate(data):
        field[i] = d
