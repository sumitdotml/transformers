"""
RoPE validation script, generated with gemini-2.5-pro-exp-03-25.
"""

import math
import torch
from ..mistral import RoPE

D_MODEL = 4
MAX_SEQ_LEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- Validation Script for RoPE running on {DEVICE} ---")

rope_module = RoPE(d_model=D_MODEL, max_seq_len=MAX_SEQ_LEN).to(DEVICE)

# --- Test 1: Identity at Position 0 ---
print("\n--- Test 1: Identity at Position 0 (offset=0, seq_len=1) ---")
input_pos0 = torch.tensor([[[1.0, 2.0, 0.5, 0.8]]], dtype=torch.float32, device=DEVICE) # B=1, S=1, D=4
rotated_pos0 = rope_module(input_pos0, offset=0)
print("Input at pos 0:\n", input_pos0)
print("Rotated at pos 0:\n", rotated_pos0)
if torch.allclose(input_pos0, rotated_pos0, atol=1e-7): # atol for float precision
    print("SUCCESS: RoPE at pos 0 is an identity transformation.")
else:
    print("FAILURE: RoPE at pos 0 is NOT an identity transformation.")
    print("Difference:\n", torch.abs(input_pos0 - rotated_pos0))

# --- Test 2: Different transformation at a different position ---
print("\n--- Test 2: Transformation at Position 1 (offset=1, seq_len=1) ---")
input_pos1 = torch.tensor([[[1.0, 2.0, 0.5, 0.8]]], dtype=torch.float32, device=DEVICE) # Same input vector
rotated_pos1 = rope_module(input_pos1, offset=1)
print("Input at pos 1 (same as pos 0 input):\n", input_pos1)
print("Rotated at pos 1:\n", rotated_pos1)
if not torch.allclose(input_pos1, rotated_pos1):
    print("SUCCESS: RoPE at pos 1 provides a different transformation than identity.")
else:
    print("FAILURE: RoPE at pos 1 IS an identity transformation (unexpected for pos 1).")
if not torch.allclose(rotated_pos0, rotated_pos1):
     print("SUCCESS: RoPE at pos 1 is different from RoPE at pos 0 for the same input vector.")
else:
     print("FAILURE: RoPE at pos 1 is THE SAME AS RoPE at pos 0 (unexpected for different positions).")

# --- Test 3: Norm preservation ---
print("\n--- Test 3: Norm Preservation ---")
input_seq = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [-1.0, -2.0, 0.0, 1.0]]], dtype=torch.float32, device=DEVICE) # B=1, S=3, D=4
rotated_seq = rope_module(input_seq, offset=0)
print("Input sequence:\n", input_seq)
print("Rotated sequence:\n", rotated_seq)

norm_input_per_token = torch.linalg.norm(input_seq, dim=-1)
norm_rotated_per_token = torch.linalg.norm(rotated_seq, dim=-1)
print("Norm of input tokens (per position):\n", norm_input_per_token)
print("Norm of rotated tokens (per position):\n", norm_rotated_per_token)

if torch.allclose(norm_input_per_token, norm_rotated_per_token, atol=1e-6):
    print("SUCCESS: Norms are preserved after RoPE application.")
else:
    print("FAILURE: Norms are NOT preserved after RoPE application.")
    print("Difference in norms (abs):\n", torch.abs(norm_input_per_token - norm_rotated_per_token))

# --- Test 4: Applying to a sequence and checking offset consistency ---
print("\n--- Test 4: Sequence Application & Offset Consistency ---")
seq_input_2tokens = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                   [5.0, 6.0, 7.0, 8.0]]], dtype=torch.float32, device=DEVICE) # B=1, S=2, D=4
rotated_seq_offset0 = rope_module(seq_input_2tokens, offset=0)
print("Original sequence:\n", seq_input_2tokens)
print("Rotated sequence (offset 0):\n", rotated_seq_offset0)
expected_token1_rotated_from_seq = rotated_seq_offset0[:, 1, :] # Second token from full rotation

token1_original_isolated = seq_input_2tokens[:, 1:2, :] # Isolate original second token (B,1,D)
print("Original second token (isolated for individual RoPE):\n", token1_original_isolated)
token1_rotated_individually_offset1 = rope_module(token1_original_isolated, offset=1)
print("Second token rotated individually with offset=1:\n", token1_rotated_individually_offset1)

if torch.allclose(expected_token1_rotated_from_seq, token1_rotated_individually_offset1.squeeze(1), atol=1e-6):
    print("SUCCESS: RoPE with offset correctly processes individual tokens as part of a sequence.")
else:
    print("FAILURE: RoPE with offset does not match sequence processing.")
    print("Difference:\n", torch.abs(expected_token1_rotated_from_seq - token1_rotated_individually_offset1.squeeze(1)))

# --- Test 5: RoPE on a known simple input and specific position (manual check) ---
print("\n--- Test 5: Known input at specific position (manual check for d_model=2) ---")
rope_d2 = RoPE(d_model=2, max_seq_len=5, base=10000.0).to(DEVICE)
simple_input_d2 = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32, device=DEVICE) # B=1, S=1, D=2

# Manual calculation for d_model=2, pos m=1 (offset=1)
# For d_model=2, d_half=1. theta_indices = [0]. theta_0 = 1.0 / (10000.0^((2*0)/2)) = 1.0.
# At pos m=1 (offset=1): angle = position_indices[1] * theta_0 = 1.0 * 1.0 = 1.0 radian.
m_theta = 1.0 
cos_m_theta = math.cos(m_theta)
sin_m_theta = math.sin(m_theta)

# Input x = [x_even, x_odd] = [1.0, 0.0]
# x_even_prime = x_even * cos_m_theta - x_odd * sin_m_theta
# x_odd_prime  = x_odd * cos_m_theta + x_even * sin_m_theta
expected_x_even_prime = 1.0 * cos_m_theta - 0.0 * sin_m_theta # = cos_m_theta
expected_x_odd_prime  = 0.0 * cos_m_theta + 1.0 * sin_m_theta # = sin_m_theta
expected_output_d2 = torch.tensor([[[expected_x_even_prime, expected_x_odd_prime]]], dtype=torch.float32, device=DEVICE)

rotated_simple_d2_pos1 = rope_d2(simple_input_d2, offset=1)
print(f"Input (d_model=2): {simple_input_d2}")
print(f"Expected output at pos 1 (manual calc): {expected_output_d2}")
print(f"Actual RoPE output at pos 1: {rotated_simple_d2_pos1}")

if torch.allclose(expected_output_d2, rotated_simple_d2_pos1, atol=1e-6):
    print("SUCCESS: RoPE matches manual calculation for d_model=2 at pos 1.")
else:
    print("FAILURE: RoPE does not match manual calculation.")
    print("Difference:\n", torch.abs(expected_output_d2 - rotated_simple_d2_pos1))

print("\n--- Validation script finished. ---") 