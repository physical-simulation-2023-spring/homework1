import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu)

@ti.data_oriented
class Transform2D:
    rotation: tm.mat2
    translation: tm.vec2
    
    @ti.func
    def ApplyToPoint(self, pt):
        return self.rotation @ pt + self.translation
    
    @ti.func
    def ApplyToPoints(self, pts, ret):
        for i in pts:
            ret[i] = self.ApplyToPoint(pts[i])

    @ti.func
    def UpdateFromTheta(self, t):
        self.rotation = tm.mat2([[tm.cos(t[None]), -tm.sin(t[None])], [tm.sin(t[None]), tm.cos(t[None])]])
    
    @ti.func
    def Inverse(self):
        ret_rotation = self.rotation.transpose()
        ret_translation = - ret_rotation @ self.translation
        return Transform2D(ret_rotation, ret_translation)
    
    
@ti.data_oriented
class Transform3D:
    rotation: tm.mat3
    translation: tm.vec3
    
    @ti.func
    def ApplyToPoint(self, pt):
        return self.rotation @ pt + self.translation
    
    @ti.func
    def ApplyToPoints(self, pts, ret):
        for i in pts:
            ret[i] = self.ApplyToPoint(pts[i])
    
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