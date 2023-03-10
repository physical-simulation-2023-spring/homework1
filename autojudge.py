from quadrotor2d import *
import numpy as np

rng = np.random.default_rng(seed=879291)

def test(to_save=False):
    global time_step
    time_step = 0.01
    p1_score = 0
    p2_score = 0
    for i in range(0, 100, 2):
        qs = {
            "transform.translation": rng.random((2, )),
            "theta": rng.random(()),
            "velocity": rng.random((2, )),
            "angular_velocity": rng.random(())
        }
        for var in qs:
            eval(var).from_numpy(qs[var])
        
        substep(True)
        for var in qs:
            qs[var] = eval(var).to_numpy()
            if to_save:
                np.save(var + str(i) + ".npy", qs[var])
            else:
                gt = np.load(var + str(i) + ".npy")
                if np.linalg.norm(qs[var] - gt) < 1e-5:
                    p1_score += 1

        qs = {
            "transform.translation": rng.random((2, )),
            "theta": rng.random(()),
            "velocity": rng.random((2, )),
            "angular_velocity": rng.random(())
        }
        for var in qs:
            eval(var).from_numpy(qs[var])
        substep(False)
        
        for var in qs:
            qs[var] = eval(var).to_numpy()
            if to_save:
                np.save(var + str(i+1) + ".npy", qs[var])
            else:
                gt = np.load(var + str(i+1) + ".npy")
                print(gt)
                if np.linalg.norm(qs[var] - gt) < 1e-5:
                    p2_score += 1
    return p1_score, p2_score

print(test(False))
