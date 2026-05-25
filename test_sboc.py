import numpy as np
import traceback
from state_based_optimal_controller import sboc

def test_sboc():
    print("Initializing sboc...")
    controller = sboc('sboc.yaml')

    # Test cases to trigger different regions of the state space
    test_cases = [
        # (name, full_state_est, w_des_radps, thr_frac)
        # full_state_est: (omega_bx, omega_by, omega_bz, phi_px, phi_nx, omega_rpx, omega_rnx)
        (
            "Nominal Hover",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, -1000.0]),
            np.array([[0.0], [0.0], [0.0]]),
            0.25
        ),
        (
            "Zero Rotor Speed",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([[0.0], [0.0], [0.0]]),
            0.25
        ),
        (
            "High Body Rates",
            np.array([5.0, -5.0, 2.0, 0.1, -0.1, 2000.0, -2000.0]),
            np.array([[0.0], [0.0], [0.0]]),
            0.5
        ),
        (
            "Extreme Rotor Speed Differential",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5000.0, -500.0]),
            np.array([[0.0], [0.0], [0.0]]),
            0.75
        ),
         (
            "Near Singularity / Small Rotor Speed",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1e-3, -1e-3]),
            np.array([[0.0], [0.0], [0.0]]),
            0.25
        )
    ]

    for name, state, w_des, thr in test_cases:
        print(f"\n--- Running Test: {name} ---")
        try:
            u = controller.get_u(state, w_des, thr)
            print(f"Success. u: {u}")
            
            # evaluate M_net
            f_px = controller.w_rapds_2_f_N.index(np.abs(state[5]))
            f_nx = controller.w_rapds_2_f_N.index(np.abs(state[6]))
            eval_args_M = tuple(state) + tuple(u) + (f_px, f_nx) + controller.params
            M_net = controller.M_net_func(*eval_args_M)
            
            print(f"M_net (Nm): \n{M_net}")
            err = state[0:3] - w_des.flatten()
            print(f"Omega error (rad/s): {err}")
            
            # Simple check: Does M_net oppose the error?
            # M_net should be roughly proportional to -error
            restoring = []
            for i in range(3):
                if abs(err[i]) > 1e-3:
                    restoring.append(np.sign(M_net[i][0]) != np.sign(err[i]))
                else:
                    restoring.append(True) # No strong error to oppose
            print(f"Is Restoring Moment? {all(restoring)} (per axis: {restoring})")
            
        except Exception as e:
            print(f"Exception triggered by test '{name}':")
            traceback.print_exc()

if __name__ == "__main__":
    test_sboc()
