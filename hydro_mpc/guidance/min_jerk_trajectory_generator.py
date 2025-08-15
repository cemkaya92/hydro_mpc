#!/usr/bin/env python3

import numpy as np

# ==================== Trajectory Generation ====================
class MinJerkTrajectoryGenerator:
    def __init__(self, a_max = np.array([0.0,0.0,0.0])):

        self._coeffs = None
        self._duration = None

        self.a_max = a_max

        self._generated = False



    # // Public Functions    
    def generate_tracking_trajectory(self, 
                            state_current: np.ndarray, 
                            state_current_target: np.ndarray):

        _p0, _v0, _a0 = state_current[0:3], state_current[3:6], state_current[6:9]
        _p0_target, _v0_target, _a0_target = state_current_target[0:3], state_current_target[3:6], state_current_target[6:9]
        _guess_tf = 5.0
        _pf_est = np.array([_p0_target[0] + _v0_target[0]*_guess_tf, 
                            _p0_target[1] + _v0_target[1]*_guess_tf, 
                            _p0_target[2] + _v0_target[2]*_guess_tf])
        
        _vf = np.array([_v0_target[0], _v0_target[1], _v0_target[2]])
        _tfs = [self._find_tf_nonzero(_p0[i], _pf_est[i], _v0[i], _vf[i], _a0[i], 0, self.a_max[i]) for i in range(3)]
        self._duration = max(_tfs)

        _pf_final = np.array([_p0_target[0] + _v0_target[0]*self._duration, _p0_target[1] + _v0_target[1]*self._duration, _p0_target[2] + _v0_target[2]*self._duration])
        self._coeffs = [self._mj_coeffs(0, self._duration, _p0[i], _pf_final[i], _v0[i], _vf[i], _a0[i], 0.0) for i in range(3)]
        
        self._generated = True

    def get_ref_at_time(self, t):

        if (not self._generated):
            return None, None, None
        
        _tau = max(0.0, t)
        _tau_clamped = min(_tau, self._duration)
        p = np.array([self._mj_eval(self._coeffs[i], _tau_clamped, 0)[0] for i in range(3)])
        v = np.array([self._mj_eval(self._coeffs[i], _tau_clamped, 0)[1] for i in range(3)])
        a = np.array([self._mj_eval(self._coeffs[i], _tau_clamped, 0)[2] for i in range(3)])

        return p, v, a

    # // Private Functions
    # Minimum Jerk Trajectory Utilities from jinays.py
    def _mj_coeffs(self, t0, tf, p0, pf, v0=0, vf=0, a0=0, af=0):
        T = tf - t0
        if T <= 1e-5:
            return np.zeros(6)
        A = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [1.0, T, T**2, T**3, T**4, T**5],
            [0.0, 1.0, 2.0*T, 3.0*T**2, 4.0*T**3, 5.0*T**4],
            [0.0, 0.0, 2.0, 6.0*T, 12.0*T**2, 20.0*T**3]
        ])
        b = np.array([p0, v0, a0, pf, vf, af])
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def _mj_eval(self,_coeffs, t, t0):
        dt = t - t0
        p = _coeffs[0] + _coeffs[1]*dt + _coeffs[2]*dt**2 + _coeffs[3]*dt**3 + _coeffs[4]*dt**4 + _coeffs[5]*dt**5
        v = _coeffs[1] + 2*_coeffs[2]*dt + 3*_coeffs[3]*dt**2 + 4*_coeffs[4]*dt**3 + 5*_coeffs[5]*dt**4
        a = 2*_coeffs[2] + 6*_coeffs[3]*dt + 12*_coeffs[4]*dt**2 + 20*_coeffs[5]*dt**3
        return p, v, a

    def _find_tf_nonzero(self,p0, pf, v0, vf, a0, af, a_max, t_min=0.1, t_max=15.0, tol=1e-3): # t_min=0.1
        _tf_low, _tf_high = t_min, t_max
        for _ in range(30):
            _tf = 0.5 * (_tf_low + _tf_high)
            if _tf <= 0:
                _tf_low = tol
                continue
            _coeffs = self._mj_coeffs(0, _tf, p0, pf, v0, vf, a0, af)
            _tvec = np.linspace(0, _tf, 50)
            _acc = np.array([self._mj_eval(_coeffs, t, 0)[2] for t in _tvec])
            if np.any(np.isnan(_acc)):
                _tf_low = _tf
                continue
            _acc_max = np.max(np.abs(_acc))
            if _acc_max > a_max:
                _tf_low = _tf
            else:
                _tf_high = _tf
            if _tf_high - _tf_low < tol:
                break
        return _tf_high
    

    
