# LLaMA2

My attempt at implementing LLaMA2.

## ToC

- [Key Formulas and Concepts](#Key-Formulas-and-Concepts)
  - [Rotary Position Embeddings (RoPE)](#RoPE)
  - [RoPE Exercises](#RoPE-Exercises)
- [References](#References)
  - [Blogs](#Blogs)
  - [Papers](#Papers)
  - [Repos](#Repos)


## Key Formulas and Concepts <a name="Key-Formulas-and-Concepts"></a>

### Rotary Position Embeddings (RoPE) <a name="RoPE"></a>

**Goal**:

Modify Query `q_m` (at position `m`) and Key `k_n` (at position `n`) into `q'_m` and `k'_n` such that their dot product `(q'_m)^T (k'_n)` depends only on the original `q_m`, `k_n`, and the relative position `m-n`.

#

**Method**:

Divide the embedding dimension `d` into `d/2` pairs. Rotate each pair `j` (dimensions `2j` and `2j+1`) by an angle that depends on the position and the pair index.

#

**Frequency $θ_j$**:

For pair `j` (where `j` goes from $0$ to $d/2 - 1$), the base frequency is $θ_j = base ^ {(-2j / d)}$. A common base is `10000`.

#

**Rotation Angle**:

For a vector at position `p` (where `p` is `m` or `n`), the rotation angle for pair `j` is $α = p * θ_j$.

#

**Rotation of a Pair**:

Given a pair $[x, y]$ (representing dimensions $2j$ and $2j+1$), the rotated pair $[x', y']$ is:

$$
x' = x * cos(α) - y * sin(α)
$$
$$
y' = x * sin(α) + y * cos(α)
$$

Or using the matrix:

$$
\begin{bmatrix}
    cos(α) & -sin(α) \\
    sin(α) & cos(α)
\end{bmatrix} * \begin{bmatrix} x \\ y \end{bmatrix}
$$

#

**Dot Product Property (for verification)**:

The dot product contribution from pair `j` is:

$$
Re[ (q_{2j} + i q_{2j+1}) * conjugate(k_{2j} + i k_{2j+1}) * e^{i (m-n) θ_j} ]
$$

where $Re$ is the real part of a complex number.

The total dot product is the sum over all pairs `j`.

Note: Conjugate of a complex number $a + bi$ is $a - bi$.

<details>
<summary><a name="RoPE-Exercises"></a>RoPE Exercises</summary>

**Exercise 1: Simple Rotation (Single Vector)**

**Setup**:

- Dimension `d` = 2 (one pair, `j`=0)
- Frequency base `base` = 100 (just for easier numbers)
- Query vector `q` = [3, 4]
- Position `m` = 1

**Tasks**:

- Calculate the frequency $θ_0$.
- Calculate the rotation angle $α = m * θ_0$.
- Calculate the rotated query vector $q'_m$. (Assume $cos(1) ≈ 0.54$, $sin(1) ≈ 0.84$ if you need numerical values, or leave as $cos(1)$, $sin(1)$).

**Exercise 2: Rotation and Dot Product (Two Vectors, d=2)**

**Setup**:

- Dimension `d` = 2 (one pair, `j`=0)
- Frequency base `base` = 100 (so $θ_0 = 1$ as in Ex 1)
- Query vector `q` = [1, 0] at position `m` = π/2
- Key vector `k` = [0, 2] at position `n` = π

**Tasks**:

- Calculate the rotation angle $α_q = m * θ_0$ for the query.
- Calculate the rotation angle $α_k = n * θ_0$ for the key.
- Calculate the rotated query $q'_m$.
- Calculate the rotated key $k'_n$.
- Calculate the dot product $(q'_m)^T (k'_n)$.

**Exercise 3: Verifying the Relative Position Property (d=2)**

**Setup**:

Use the same `d=2`, `base=100`, `θ_0=1`, `q=[1, 0]`, `k=[0, 2]` as in Exercise 2.

In Exercise 2, `m = π/2`, `n = π`, so the relative position `m - n = -π/2`.

Now, choose new positions `m' = 0` and `n' = π/2`. Note that `m' - n' = -π/2`, the same relative position.

**Tasks**:

- Calculate the rotation angle $α'_q = m' * θ_0$ for the query at `m'`.
- Calculate the rotation angle $α'_k = n' * θ_0$ for the key at `n'`.
- Calculate the rotated query $q'_{m'}$.
- Calculate the rotated key $k'_{n'}$.
- Calculate the dot product $(q'_{m'})^T (k'_{n'})$
- Compare this dot product to the one you calculated in Exercise 2. Are they the same? (They should be!)

**Exercise 4: Higher Dimension Rotation (d=4)**

**Setup**:

- Dimension `d` = 4 (two pairs, `j`=0 and `j`=1)
- Frequency base `base` = 4 (to get nice angles)
- Query vector `q` = [1, 1, 2, 0]
- Position `m` = π

**Tasks**:

- Calculate the frequencies $θ_0$ and $θ_1$ using `base=4`, `d=4`.
- Calculate the rotation angle $α_0 = m * θ_0$ for the first pair (`q_0`, `q_1`).
- Calculate the rotation angle $α_1 = m * θ_1$ for the second pair (`q_2`, `q_3`).
- Rotate the first pair [1, 1] using angle $α_0$ to get [q'_0, q'_1].
- Rotate the second pair [2, 0] using angle $α_1$ to get [q'_2, q'_3].
- Combine the results to form the full rotated vector $q'_m = [q'_0, q'_1, q'_2, q'_3]$.

**Exercise 5: Dot Product with Higher Dimension (d=4)**

**Setup**:

Use the same `d=4`, `base=4`, $θ_0$, $θ_1$ as in Exercise 4.

- Query vector `q` = [1, 1, 2, 0] at position `m` = π. (You already calculated $q'_m$ in Exercise 4).
- Key vector `k` = [1, 0, 0, 1] at position `n` = 0.

**Tasks**:

- Calculate the rotation angle $α_{k0} = n * θ_0$ for the first pair of `k`.
- Calculate the rotation angle $α_{k1} = n * θ_1$ for the second pair of `k`.
- Rotate the first pair of `k`, [1, 0], using $α_{k0}$ to get [k'_0, k'_1]. (Hint: What happens when the angle is 0?)
- Rotate the second pair of `k`, [0, 1], using $α_{k1}$ to get [k'_2, k'_3].
- Form the full rotated key vector $k'_n = [k'_0, k'_1, k'_2, k'_3]$.
- Retrieve the rotated query $q'_m$ from Exercise 4.
- Calculate the final dot product $(q'_m)^T (k'_n) = q'_0*k'_0 + q'_1*k'_1 + q'_2*k'_2 + q'_3*k'_3$.

</details>

## References <a name="References"></a>

### Blogs <a name="Blogs"></a>

- [LLaMA-2 from the Ground Up by Cameron R. Wolfe](https://substack.com/home/post/p-135824233)
- [Rotary Position Embeddings by EleutherAI](https://blog.eleuther.ai/rotary-embeddings/)
- [The Practical Guide to LLMs - LLaMA-2 by Georgian Impact Blog](https://medium.com/georgian-impact-blog/the-practical-guide-to-llms-llama-2-cdf21d540ce3)

### Papers <a name="Papers"></a>

- [LLaMA2 Paper](https://arxiv.org/abs/2402.12578)
- [Group Query Attention Paper](https://arxiv.org/abs/2305.13245)
- [RMSNorm Paper](https://arxiv.org/abs/1910.07467)
- [SwiGLU Paper](https://arxiv.org/abs/2002.05202)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)

### Repos <a name="Repos"></a>

- [LLaMA Cookbook](https://github.com/meta-llama/llama-cookbook)
- [Official LLaMA2 Implementation](https://github.com/facebookresearch/llama/blob/main/llama/model.py)
- [Georgian Impact Blog's LLaMA2 Implementation](https://github.com/georgian-io/LLM-Finetuning-Toolkit/tree/main/llama2)
