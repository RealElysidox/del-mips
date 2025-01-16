#![allow(non_snake_case)]

use del_geo_core::mat3_col_major::Mat3ColMajor;
use del_geo_core::mat3_row_major::*;
use del_geo_core::vec3::*;
use std::f64;

type Mat3 = [f64; 9];

fn skew(v: &[f64; 3]) -> Mat3 {
    [0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0]
}

fn deformation_gradient_of_tet(
    pos0: &[f64; 3],
    pos1: &[f64; 3],
    pos2: &[f64; 3],
    pos3: &[f64; 3],
    Pos0: &[f64; 3],
    Pos1: &[f64; 3],
    Pos2: &[f64; 3],
    Pos3: &[f64; 3],
) -> Mat3 {
    let dx10 = pos1.sub(&pos0);
    let dx20 = pos2.sub(&pos0);
    let dx30 = pos3.sub(&pos0);

    let dX10 = Pos1.sub(&Pos0);
    let dX20 = Pos2.sub(&Pos0);
    let dX30 = Pos3.sub(&Pos0);

    let mat_x = [
        dx10[0], dx20[0], dx30[0], dx10[1], dx20[1], dx30[1], dx10[2], dx20[2], dx30[2],
    ];
    let mat_X = [
        dX10[0], dX20[0], dX30[0], dX10[1], dX20[1], dX30[1], dX10[2], dX20[2], dX30[2],
    ];

    mat_x.mult_mat_row_major(&mat_X.try_inverse().unwrap())
}

fn diff_deformation_gradient_of_tet(
    pos0: &[f64; 3],
    pos1: &[f64; 3],
    pos2: &[f64; 3],
    pos3: &[f64; 3],
) -> [[f64; 3]; 4] {
    let mut bi0 = [
        pos1[0] - pos0[0],
        pos2[0] - pos0[0],
        pos3[0] - pos0[0],
        pos1[1] - pos0[1],
        pos2[1] - pos0[1],
        pos3[1] - pos0[1],
        pos1[2] - pos0[2],
        pos2[2] - pos0[2],
        pos3[2] - pos0[2],
    ];
    bi0 = bi0.try_inverse().unwrap();
    let mut df = [[0.0; 3]; 4];
    df[0][0] = -bi0[0] - bi0[3] - bi0[6];
    df[0][1] = -bi0[1] - bi0[4] - bi0[7];
    df[0][2] = -bi0[2] - bi0[5] - bi0[8];
    df[1][0] = bi0[0];
    df[1][1] = bi0[1];
    df[1][2] = bi0[2];
    df[2][0] = bi0[3];
    df[2][1] = bi0[4];
    df[2][2] = bi0[5];
    df[3][0] = bi0[6];
    df[3][1] = bi0[7];
    df[3][2] = bi0[8];
    df
}

pub fn diff_piola_kirchhoff1st(
    u0: &Mat3,
    s0: &Mat3,
    v0: &Mat3,
    diff: &[[[f64; 3]; 3]; 9],
    neohook: &dyn Fn(f64, f64, f64) -> (f64, [f64; 3], [[f64; 3]; 3]),
) -> (f64, Mat3, [[[[f64; 3]; 3]; 3]; 3]) {
    let (w0, dwdl0, ddwddl0) = neohook(s0[0], s0[4], s0[8]);
    let t0 = [dwdl0[0], 0.0, 0.0, 0.0, dwdl0[1], 0.0, 0.0, 0.0, dwdl0[2]];
    let p0 = u0.mult_mat_row_major(&t0.mult_mat_row_major(&transpose(&v0)));
    let mut dpdf = [[[[0.0; 3]; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let du =
                u0.mult_mat_row_major(&skew(&[-diff[0][i][j], -diff[1][i][j], -diff[2][i][j]]));
            let dv =
                v0.mult_mat_row_major(&skew(&[-diff[6][i][j], -diff[7][i][j], -diff[8][i][j]]));
            let ds = Mat3::from_diagonal(&[diff[3][i][j], diff[4][i][j], diff[5][i][j]]);

            let t0_0 = ddwddl0[0][0] * ds[0] + ddwddl0[0][1] * ds[4] + ddwddl0[0][2] * ds[8];
            let t0_1 = ddwddl0[1][0] * ds[0] + ddwddl0[1][1] * ds[4] + ddwddl0[1][2] * ds[8];
            let t0_2 = ddwddl0[2][0] * ds[0] + ddwddl0[2][1] * ds[4] + ddwddl0[2][2] * ds[8];

            let dt = [t0_0, 0.0, 0.0, 0.0, t0_1, 0.0, 0.0, 0.0, t0_2];

            let dp = du
                .mult_mat_row_major(&t0.mult_mat_row_major(&transpose(&v0)))
                .add(
                    &u0.mult_mat_row_major(&dt.mult_mat_row_major(&transpose(&v0)))
                        .add(&u0.mult_mat_row_major(&(t0.mult_mat_row_major(&transpose(&dv))))),
                );
            for k in 0..3 {
                for l in 0..3 {
                    dpdf[k][l][i][j] = dp[3 * k + l];
                }
            }
        }
    }
    (w0, p0, dpdf)
}

pub fn wdwddw_invertible_fem(
    pos: &[[f64; 3]; 4],
    pos_ref: &[[f64; 3]; 4],
    neohook: &dyn Fn(f64, f64, f64) -> (f64, [f64; 3], [[f64; 3]; 3]),
) -> (f64, [[f64; 3]; 4], [[[[f64; 3]; 3]; 4]; 4]) {
    let mut dw = [[0.0; 3]; 4];
    let mut ddw = [[[[0.0; 3]; 3]; 4]; 4];
    let f0 = deformation_gradient_of_tet(
        &pos[0],
        &pos[1],
        &pos[2],
        &pos[3],
        &pos_ref[0],
        &pos_ref[1],
        &pos_ref[2],
        &pos_ref[3],
    );

    let dfdu = diff_deformation_gradient_of_tet(&pos_ref[0], &pos_ref[1], &pos_ref[2], &pos_ref[3]);

    let (u0, s0, v0) = svd(&f0, 30);
    let diff = svd_differential(u0, s0, v0);
    let (w0, p0, dpdf) = diff_piola_kirchhoff1st(&u0, &s0, &v0, &diff, neohook);

    for ino in 0..4 {
        for i in 0..3 {
            dw[ino][i] = p0[3 * i] * dfdu[ino][0]
                + p0[3 * i + 1] * dfdu[ino][1]
                + p0[3 * i + 2] * dfdu[ino][2];
        }
        for jno in 0..4 {
            for i in 0..3 {
                for j in 0..3 {
                    ddw[ino][jno][i][j] = dpdf[i][0][j][0] * dfdu[ino][0] * dfdu[jno][0]
                        + dpdf[i][0][j][1] * dfdu[ino][0] * dfdu[jno][1]
                        + dpdf[i][0][j][2] * dfdu[ino][0] * dfdu[jno][2]
                        + dpdf[i][1][j][0] * dfdu[ino][1] * dfdu[jno][0]
                        + dpdf[i][1][j][1] * dfdu[ino][1] * dfdu[jno][1]
                        + dpdf[i][1][j][2] * dfdu[ino][1] * dfdu[jno][2]
                        + dpdf[i][2][j][0] * dfdu[ino][2] * dfdu[jno][0]
                        + dpdf[i][2][j][1] * dfdu[ino][2] * dfdu[jno][1]
                        + dpdf[i][2][j][2] * dfdu[ino][2] * dfdu[jno][2];
                }
            }
        }
    }
    (w0, dw, ddw)
}
