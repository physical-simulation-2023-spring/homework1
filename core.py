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