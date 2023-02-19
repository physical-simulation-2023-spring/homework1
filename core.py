import taichi as ti
import taichi.math as tm
import numpy as np
ti.init(arch=ti.cpu)

link_list = []
joint_list = []

@ti.func
def AddLink(link_name):
    if link_name == "hinge":
        pass
    elif link_name == "?":
        pass


@ti.func
def AddJoint():
    pass


@ti.func
def Not():
    pass