import pytest
import numpy as np
import requests
from requests.exceptions import InvalidJSONError
from src.quantum.middleware.minimizer import MinimizerCommand

# --- THE FIX: Include stdgates.inc for QASM 3.0 ---
QAOA_2Q_QASM = """
OPENQASM 3.0;
include "stdgates.inc"; 

// --- DECLARE INPUTS (The Missing Piece) ---
input float _gammas_0_;
input float _betas_0_;

qubit[2] q;
bit[2] c;

h q[0];
h q[1];

rz(_gammas_0_) q[0]; 
rz(_gammas_0_) q[1];

rx(_betas_0_) q[0];
rx(_betas_0_) q[1];

c = measure q;
"""

QAOA_3Q_LINE_QASM = """
OPENQASM 3.0;
include "stdgates.inc";

input float g0; input float g1; input float g2; input float g3;
input float b0; input float b1; input float b2; input float b3;

qubit[3] q;
bit[3] c;

h q[0]; h q[1]; h q[2];

// Layer 1
cx q[0], q[1]; rz(g0) q[1]; cx q[0], q[1];
cx q[1], q[2]; rz(g0) q[2]; cx q[1], q[2];
rz(g0) q[0]; rz(g0) q[1]; rz(g0) q[2];
rx(b0) q[0]; rx(b0) q[1]; rx(b0) q[2];

// --- LAYER 2 ---
cx q[0], q[1]; rz(g1) q[1]; cx q[0], q[1];
cx q[1], q[2]; rz(g1) q[2]; cx q[1], q[2];
rz(g1) q[0]; rz(g1) q[1]; rz(g1) q[2];
rx(b1) q[0]; rx(b1) q[1]; rx(b1) q[2];

// --- LAYER 3 ---
cx q[0], q[1]; rz(g2) q[1]; cx q[0], q[1];
cx q[1], q[2]; rz(g2) q[2]; cx q[1], q[2];
rz(g2) q[0]; rz(g2) q[1]; rz(g2) q[2];
rx(b2) q[0]; rx(b2) q[1]; rx(b2) q[2];

// --- LAYER 4 ---
cx q[0], q[1]; rz(g3) q[1]; cx q[0], q[1];
cx q[1], q[2]; rz(g3) q[2]; cx q[1], q[2];
rz(g3) q[0]; rz(g3) q[1]; rz(g3) q[2];
rx(b3) q[0]; rx(b3) q[1]; rx(b3) q[2];

c = measure q;"""

class TestMinimizerStack:

    @pytest.fixture
    def minimizer(self):
        """Initializes the command middleware."""
        # Ensure your MinimizerCommand points to the correct URL internally
        return MinimizerCommand(
            shots=2048, 
            seed=42
        )

    def test_gpu_server_connectivity(self, minimizer):
        """Step 1: Verify the server is reachable."""
        try:
            # FastAPI typically redirects /health/ to /health, 
            # but we check the base to be safe
            response = requests.get("http://127.0.0.1:8000/health")
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.fail("GPU Server is not running at 127.0.0.1:8000")

    def test_gradient_sensitivity(self, minimizer):
        """Step 2: Verify the finite-difference gradient isn't 'blind'."""
        qubo_dict = {(0, 0): 1.0, (1, 1): 1.0}
        test_angles = np.array([np.pi/4, np.pi/4])
        
        # Use the QASM with stdgates.inc
        grad = minimizer._gradient_batched(
            test_angles, 
            QAOA_2Q_QASM, 
            qubo_dict, 
            0.0
        )
        
        norm = np.linalg.norm(grad)
        print(f"\n[DEBUG] Measured Gradient: {grad} | Norm: {norm}")
        
        assert norm > 1e-4, "Gradient is zero. Check epsilon or shot noise."

    def test_json_nan_protection(self, minimizer):
        """
        Step 3: Check NaN protection.
        Note: requests.post throws InvalidJSONError if it sees a NaN.
        """
        bad_angles = np.array([np.nan, 0.5])
        qubo = {(0,0): 1.0}
        
        # 1. Call the function - it should NOT raise an exception now!
        energy = minimizer._cost_function(bad_angles, "FAKE_QASM", qubo, 0.0)
        
        # 2. Assert that it returned your 'High Penalty' value
        assert energy == 1e6, f"Expected penalty 1e6 for NaN angles, got {energy}"

    def test_parameter_binding(self, minimizer):
        """Step 4: Verify energy changes with angles (Binding check)."""
        qubo_dict = {(0, 0): 1.0, (1, 1): 1.0}
        
        e1 = minimizer._cost_function(np.array([0.1, 0.1]), QAOA_2Q_QASM, qubo_dict, 0.0)
        e2 = minimizer._cost_function(np.array([1.5, 1.5]), QAOA_2Q_QASM, qubo_dict, 0.0)
        
        print(f"\n[DEBUG] E1: {e1:.6f} | E2: {e2:.6f}")
        assert e1 != e2, "Energy is static. Parameters are NOT binding."
        
    def test_global_batch_one_hot(self, minimizer: MinimizerCommand):
        """
        Verifies that global sampling finds a better starting point 
        than a random guess for a 'Pick One' constraint.
        """
        # QUBO designed to have minima at 01 and 10
        qubo_dict = {(0, 0): 2.0, (1, 1): 2.0, (0, 1): -4.0}
        constant = 0.0
        p = 1
        
        # We use the QASM string from your previous tests
        qasm_str = QAOA_2Q_QASM 

        # Run the global batched minimizer
        res = minimizer.minimize_cost_global_batched(
            qasm_str, 
            qubo_dict, 
            constant, 
            p
        )

        print(f"\n[DEBUG] Global Best Energy: {res.fun}")
        print(f"[DEBUG] Optimal Angles: {res.x}")

        # Assertions
        assert res.status in [0, 2], f"Optimizer crashed: {res.message}"
        assert res.fun < 0.005, f"Energy {res.fun} is too high"    
        # assert res.success or res.status == 0, "Optimization did not converge"
        assert not np.isnan(res.x).any(), "Result contains NaNs"

    def test_qaoa_high_depth_p4(self, minimizer: MinimizerCommand):
        """
        Tests QAOA at p=4 for a 2-qubit 'Pick One' QUBO.
        Higher p should converge faster or stay at the global minimum.
        """
        qubo_dict = {(0, 0): 2.0, (1, 1): 2.0, (0, 1): -4.0}
        constant = 0.0
        p = 4  # 8 parameters total
        
        qasm_str = QAOA_2Q_QASM # Ensure your QASM generator supports p layers!

        res = minimizer.minimize_cost_global_batched(
            qasm_str, 
            qubo_dict, 
            constant, 
            p
        )

        print(f"\n[DEBUG] P=4 Best Energy: {res.fun}")
        print(f"[DEBUG] P=4 Optimal Angles: {res.x.round(3)}")

        assert res.fun < 0.005
        assert len(res.x) == 8
        
    def test_hard_maxcut_p4(self, minimizer: MinimizerCommand):
        """
        3-Qubit Line Max-Cut (0-1-2). 
        Known Minima: '101' or '010' with Energy -2.0.
        """
        # 1. Setup the 'Hard' problem
        qubo_dict = {(0, 1): 2.0, (1, 2): 2.0, (0, 0): -1.0, (1, 1): -2.0, (2, 2): -1.0}
        constant = 0.0
        p = 4
        
        # You'll need a 3-qubit QAOA QASM string here
        qasm_str = QAOA_3Q_LINE_QASM 

        # 2. Increase sampling density for 8D space
        minimizer.global_samples = 2048 
        
        # 3. Run optimization
        res = minimizer.minimize_cost_global_batched(
            qasm_str, 
            qubo_dict, 
            constant, 
            p
        )

        print(f"\n[DEBUG] Hard Test Energy: {res.fun}")
        print(f"[DEBUG] Optimal Bitstring (Approx): {res.x}")

        # 4. Assertions
        # With p=4, we expect to get very close to the theoretical -2.0
        assert res.fun <= -1.95, f"Failed to find global Max-Cut. Energy {res.fun} only found local minimum."
        assert res.status in [0, 2]