use crate::utils::*;

pub fn mat3_3base_of_tet(
    p1: &[f64; 3],
    p2: &[f64; 3],
    p3: &[f64; 3],
    p4: &[f64; 3],
) -> [[f64; 3]; 3] {
    let v1 = sub_vec(p2, p1);
    let v2 = sub_vec(p3, p1);
    let v3 = sub_vec(p4, p1);
    transpose_mat3x3(&[v1, v2, v3])
}

pub fn deformation_gradient_of_tet(p: &[[f64; 3]; 4], q: &[[f64; 3]; 4]) -> [[f64; 3]; 3] {
    let p_base = mat3_3base_of_tet(&p[0], &p[1], &p[2], &p[3]);
    let q_base = mat3_3base_of_tet(&q[0], &q[1], &q[2], &q[3]);
    let p_base_inv = inverse_mat3(&p_base);
    mul_mat3x3(&q_base, &p_base_inv)
}

pub fn diff_deformation_gradient_of_tet(
    p0: &[f64; 3],
    p1: &[f64; 3],
    p2: &[f64; 3],
    p3: &[f64; 3],
) -> [[f64; 3]; 4] {
    let mut bi0 = [sub_vec(p1, p0), sub_vec(p2, p0), sub_vec(p3, p0)];
    bi0 = transpose_mat3x3(&bi0);
    bi0 = inverse_mat3(&bi0);
    [
        [
            -bi0[0][0] - bi0[1][0] - bi0[2][0],
            -bi0[0][1] - bi0[1][1] - bi0[2][1],
            -bi0[0][2] - bi0[1][2] - bi0[2][2],
        ],
        bi0[0],
        bi0[1],
        bi0[2],
    ]
}
