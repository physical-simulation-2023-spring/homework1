import taichi as ti
import taichi.math as tm
import numpy as np
ti.init(arch=ti.cpu)

link_list = []
joint_list = []

# Some math utils.
@ti.func
def InverseTransform(transform):
    pass

@ti.func
def ApplyTransformToPoint(transform, point):
    pass

@ti.func
def GetRotMatrix2D(omega):
    pass

@ti.func
def GetRotMatrixDerivative2D(omega):
    pass

@ti.func
def GetRotMatrixSecondDerivative2D(omega):
    pass

