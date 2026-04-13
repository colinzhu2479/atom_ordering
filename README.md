# Motivation

We intend to introduce a consistent framework for description of molecular fragment geometries that arise from different molecular systems. Such a descriptor should allow for translational, rotational, and permutational invariance within the fragments, so that chemically identical fragments derived from different systems, or such fragments derived from different physical regions of the same system, have the same consistent mathematical description and thus can be compared with each other.

When training ML models on molecular data, the same physical configuration can appear in many equivalent representations depending on how the atoms are ordered or how the molecule is oriented in space. This module removes that ambiguity by:

1. Sorting atoms by atomic number for a consistent ordering
2. Centering coordinates on the center of mass
3. Rotating the fragment into its principal inertia axis frame
4. Resolving reflection ambiguity by aligning against a reference geometry

The result is a canonical representation where two physically identical configurations always map to the same input vector.

More details can be found in:
Xiao Zhu, S. S. Iyengar. Large Language Model-Type Architecture for High-Dimensional Molecular Potential Energy Surfaces. Physics Review X ,16 , 011012 (2026). DOI: https://doi.org/10.1103/2qcy-8n8g

# fragment_transform

A Python module for canonicalizing molecular fragment geometries and forces by aligning them to a consistent reference frame based on the principal axes of inertia. This is primarily used as a preprocessing step for machine learning models of molecular potential energy surfaces and atomic forces.


## Dependencies

- Python 3.x
- NumPy
- SciPy
- matplotlib

## API Reference

### `atom_mass_mapping(atom_num)`

Maps atomic numbers to approximate atomic masses for elements 1–20.

**Arguments:**
- `atom_num` — array of integer atomic numbers, shape `(n,)`

**Returns:**
- array of atomic masses, shape `(n,)`

**Example:**
```python
atom_num = np.array([8, 1, 1])   # water: O, H, H
masses = atom_mass_mapping(atom_num)
# → array([16., 1., 1.])
```

---

### `center_of_mass_xyz(xyz, atom_mass)`

Translates coordinates so the center of mass is at the origin.

**Arguments:**
- `xyz` — atomic coordinates, shape `(n, 3)`
- `atom_mass` — atomic masses, shape `(n,)`

**Returns:**
- `xyz_centered` — translated coordinates, shape `(n, 3)`
- `c_m` — original center of mass vector, shape `(3,)` (needed to invert the transform)

---

### `moment_of_innertia_axis(xyz, atom_mass)`

Computes the principal axes of the inertia tensor via eigendecomposition.

**Arguments:**
- `xyz` — mass-centered coordinates, shape `(n, 3)`
- `atom_mass` — atomic masses, shape `(n,)`

**Returns:**
- `v` — rotation matrix, shape `(3, 3)`, where rows are principal axis vectors ordered by descending eigenvalue. `v[0]` is the axis of largest moment of inertia.

---

### `permute_by_type_and_dis(xyz, force, atom_num, atom_mass)`

Sorts all per-atom arrays by ascending atomic number, establishing a canonical atom ordering.

**Arguments:**
- `xyz` — coordinates, shape `(n, 3)`
- `force` — forces, shape `(n, 3)`
- `atom_num` — atomic numbers, shape `(n,)`
- `atom_mass` — atomic masses, shape `(n,)`

**Returns:**
- `xyz`, `force`, `atom_num`, `atom_mass` — sorted arrays
- `order` — integer index array that produced the sort, shape `(n,)` (needed to invert the permutation)

---

### `transform_fragment(xyz, force, atom_num, atom_mass, target_direction, ref_xyz)`

Full single-direction canonical transform: sorts atoms, centers on center of mass, rotates into principal axis frame, and resolves reflection ambiguity along one axis using a reference geometry.

**Arguments:**
- `xyz` — coordinates, shape `(n, 3)`
- `force` — forces, shape `(n, 3)`. Pass a dummy array (e.g. `ref_xyz`) if forces are not needed.
- `atom_num` — atomic numbers, shape `(n,)`
- `atom_mass` — atomic masses, shape `(n,)`
- `target_direction` — which principal axis to align: `'x'`, `'y'`, or `'z'`
- `ref_xyz` — reference geometry used to resolve reflection, shape `(n, 3)`. Use `np.ones([n, 3])` as a neutral placeholder for the first structure in a dataset.

**Returns:**
- `xyz_t` — transformed coordinates, shape `(n, 3)`
- `force_t` — transformed forces, shape `(n, 3)`
- `atom_num` — sorted atomic numbers, shape `(n,)`
- `v` — principal axis rotation matrix, shape `(3, 3)`
- `order` — permutation index array, shape `(n,)`

**Example:**
```python
import numpy as np
import fragment_transform

xyz = np.array([
    [-2.152, 1.936, 1.384],
    [-2.481, 1.959, 2.293],
    [-2.608, 2.639, 0.903],
])
atom_num  = np.array([8, 1, 1])
atom_mass = fragment_transform.atom_mass_mapping(atom_num)
ref       = np.ones([3, 3])

xyz_t, force_t, atom_num_t, v, order = fragment_transform.transform_fragment(
    xyz, ref, atom_num, atom_mass, target_direction='z', ref_xyz=ref)
```

---

### `global_transform_fragment(xyz, force, atom_num, atom_mass, ref_xyz)`

Generator that applies `transform_fragment` along all three principal axes (`x`, `y`, `z`) in sequence. Used when reconstructing all three force components independently.

**Arguments:** same as `transform_fragment` except no `target_direction` (all three are yielded).

**Yields** (once per direction, 3 total):
- `xyz_t`, `force_t`, `atom_num`, `v`, `order` — same as `transform_fragment` returns

**Example:**
```python
reconstructed_force = np.zeros((n, 3))

for c, (xyz_t, force_t, atom_num_t, v, order) in enumerate(
        fragment_transform.global_transform_fragment(xyz, force, atom_num, atom_mass, ref)):
    reconstructed_force[:, c] = force_t[:, c][np.argsort(order)]
```

**Reconstructing xyz** — since `c_m` is not yielded (it is internal to the generator), compute it externally before calling if you need to invert the transform:

```python
xyz_s, force_s, atom_num_s, atom_mass_s, _ = fragment_transform.permute_by_type_and_dis(
    xyz.copy(), force.copy(), atom_num.copy(), atom_mass.copy())
_, c_m = fragment_transform.center_of_mass_xyz(xyz_s, atom_mass_s)

for xyz_t, force_t, atom_num_t, v, order in fragment_transform.global_transform_fragment(...):
    recovered_xyz = (np.linalg.inv(v) @ xyz_t[np.argsort(order)].T).T + c_m
```

---

## Inverting the Transform

To recover the original coordinates and forces from a transformed output:

```python
# xyz: invert rotation then restore center of mass
recovered_xyz = (np.linalg.inv(v) @ xyz_t[np.argsort(order)].T).T + c_m

# force: invert rotation only (forces are translation-invariant)
recovered_force = (np.linalg.inv(v) @ force_t[np.argsort(order)].T).T
```

Note that `c_m` must be computed externally before calling `global_transform_fragment` (see above).

---

## Running Tests

```bash
cd <project_root>
python3 -m pytest tests/test_fragment_transform.py -v
```

Tests cover: center of mass centering and invertibility, inertia axis orthonormality and eigenvalue ordering, atom sorting, all three transform directions, xyz and force reconstruction, and permutation invariance of both `transform_fragment` and `global_transform_fragment`.
