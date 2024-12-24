use nalgebra::Matrix3;
use nalgebra::Vector3;
use std::f64;

type Mat3 = Matrix3<f64>;
type Vec3 = Vector3<f64>;

fn skew(v: &[f64; 3]) -> Mat3 {
    Mat3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

fn svd3<
    T: From<f64>
        + Copy
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialOrd<T>
        + std::ops::Neg<Output = T>,
>(
    a: Mat3,
    max_iter: usize,
) -> (Mat3, Mat3, Mat3) {
    let svd = nalgebra::SVD::new(a, true, true);
    let u = svd.u.unwrap();
    let s = svd.singular_values;
    let v_t = svd.v_t.unwrap();

    let s_mat = Mat3::new(s[0], 0.0, 0.0, 0.0, s[1], 0.0, 0.0, 0.0, s[2]);
    (u, s_mat, v_t.transpose())
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
    let x0 = Vec3::new(pos0[0], pos0[1], pos0[2]);
    let x1 = Vec3::new(pos1[0], pos1[1], pos1[2]);
    let x2 = Vec3::new(pos2[0], pos2[1], pos2[2]);
    let x3 = Vec3::new(pos3[0], pos3[1], pos3[2]);

    let X0 = Vec3::new(Pos0[0], Pos0[1], Pos0[2]);
    let X1 = Vec3::new(Pos1[0], Pos1[1], Pos1[2]);
    let X2 = Vec3::new(Pos2[0], Pos2[1], Pos2[2]);
    let X3 = Vec3::new(Pos3[0], Pos3[1], Pos3[2]);

    let dx10 = x1 - x0;
    let dx20 = x2 - x0;
    let dx30 = x3 - x0;

    let dX10 = X1 - X0;
    let dX20 = X2 - X0;
    let dX30 = X3 - X0;

    let mat_x = Mat3::from_columns(&[dx10, dx20, dx30]);
    let mat_X = Mat3::from_columns(&[dX10, dX20, dX30]);

    mat_x * mat_X.try_inverse().unwrap()
}

fn diff_deformation_gradient_of_tet(
    dfdu: &mut [[f64; 3]; 4],
    pos0: &[f64; 3],
    pos1: &[f64; 3],
    pos2: &[f64; 3],
    pos3: &[f64; 3],
) {
    let X0 = Vec3::new(pos0[0], pos0[1], pos0[2]);
    let X1 = Vec3::new(pos1[0], pos1[1], pos1[2]);
    let X2 = Vec3::new(pos2[0], pos2[1], pos2[2]);
    let X3 = Vec3::new(pos3[0], pos3[1], pos3[2]);

    let dX10 = X1 - X0;
    let dX20 = X2 - X0;
    let dX30 = X3 - X0;
    let mat_X = Mat3::from_columns(&[dX10, dX20, dX30]);
    let mat_X_inv = mat_X.try_inverse().unwrap();

    let mut vec = [Vec3::zeros(); 4];
    vec[0][0] = -1.0;
    vec[0][1] = -1.0;
    vec[0][2] = -1.0;
    vec[1][0] = 1.0;
    vec[2][1] = 1.0;
    vec[3][2] = 1.0;
    for i in 0..4 {
        let v = mat_X_inv * vec[i];
        dfdu[i][0] = v[0];
        dfdu[i][1] = v[1];
        dfdu[i][2] = v[2];
    }
}

fn svd3_differential(diff: &mut [[[[f64; 3]; 3]; 3]; 9], u0: Mat3, s0: Mat3, v0: Mat3) {
    let mut e = [[0.0; 3]; 3];
    for i in 0..3 {
        e[i][i] = 1.0;
    }

    for i in 0..3 {
        for j in 0..3 {
            let du = -u0 * skew(&[0.0, 0.0, 0.0]);
            let dv = -v0 * skew(&[0.0, 0.0, 0.0]);
            let ds = Mat3::from_diagonal(&Vector3::new(0.0, 0.0, 0.0));

            for k in 0..3 {
                for l in 0..3 {
                    let mut v = [0.0; 3];
                    if k == i && l == j {
                        v[i] = 1.0;
                    }

                    let mut du = -u0 * skew(&[0.0, 0.0, 0.0]);
                    let mut dv = -v0 * skew(&[0.0, 0.0, 0.0]);
                    let mut ds = Mat3::from_diagonal(&Vector3::new(0.0, 0.0, 0.0));
                    if i < 3 && j < 3 {
                        du = -u0
                            * skew(&[
                                if j == 1 { 1.0 } else { 0.0 } * e[k][0],
                                if j == 2 { 1.0 } else { 0.0 } * e[k][1],
                                if j == 0 { 1.0 } else { 0.0 } * e[k][2],
                            ]);
                        dv = -v0
                            * skew(&[
                                if j == 1 { 1.0 } else { 0.0 } * e[k][0],
                                if j == 2 { 1.0 } else { 0.0 } * e[k][1],
                                if j == 0 { 1.0 } else { 0.0 } * e[k][2],
                            ]);
                        ds = Mat3::from_diagonal(&Vector3::new(
                            if k == 0 { 1.0 } else { 0.0 },
                            if k == 1 { 1.0 } else { 0.0 },
                            if k == 2 { 1.0 } else { 0.0 },
                        ));
                    }
                    diff[0][i][j][k] = du[(0, 0)];
                    diff[1][i][j][k] = du[(1, 0)];
                    diff[2][i][j][k] = du[(2, 0)];
                    diff[3][i][j][k] = ds[(0, 0)];
                    diff[4][i][j][k] = ds[(1, 1)];
                    diff[5][i][j][k] = ds[(2, 2)];
                    diff[6][i][j][k] = dv[(0, 0)];
                    diff[7][i][j][k] = dv[(1, 0)];
                    diff[8][i][j][k] = dv[(2, 0)];
                }
            }
        }
    }
}

fn diff_piola_kirchhoff1st(
    p0: &mut Mat3,
    dpdf: &mut [[[[f64; 3]; 3]; 3]; 3],
    u0: Mat3,
    s0: Mat3,
    v0: Mat3,
    diff: &[[[[f64; 3]; 3]; 3]; 9],
    neohook: &dyn Fn(&mut [f64; 3], &mut [[f64; 3]; 3], f64, f64, f64) -> f64,
) -> f64 {
    let mut dw_dl0 = [0.0; 3];
    let mut ddw_ddl0 = [[0.0; 3]; 3];
    let w0 = neohook(
        &mut dw_dl0,
        &mut ddw_ddl0,
        s0[(0, 0)],
        s0[(1, 1)],
        s0[(2, 2)],
    );
    let t0 = Mat3::new(
        dw_dl0[0], 0.0, 0.0, 0.0, dw_dl0[1], 0.0, 0.0, 0.0, dw_dl0[2],
    );
    *p0 = u0 * (t0 * v0.transpose());

    for i in 0..3 {
        for j in 0..3 {
            let du = -u0 * skew(&[diff[0][i][j][0], diff[1][i][j][1], diff[2][i][j][2]]);
            let dv = -v0 * skew(&[diff[6][i][j][0], diff[7][i][j][1], diff[8][i][j][2]]);
            let ds = Mat3::from_diagonal(&Vector3::new(
                diff[3][i][j][0],
                diff[4][i][j][1],
                diff[5][i][j][2],
            ));

            let t0_0 = ddw_ddl0[0][0] * ds[(0, 0)]
                + ddw_ddl0[0][1] * ds[(1, 1)]
                + ddw_ddl0[0][2] * ds[(2, 2)];
            let t0_1 = ddw_ddl0[1][0] * ds[(0, 0)]
                + ddw_ddl0[1][1] * ds[(1, 1)]
                + ddw_ddl0[1][2] * ds[(2, 2)];
            let t0_2 = ddw_ddl0[2][0] * ds[(0, 0)]
                + ddw_ddl0[2][1] * ds[(1, 1)]
                + ddw_ddl0[2][2] * ds[(2, 2)];

            let dt = Mat3::new(t0_0, 0.0, 0.0, 0.0, t0_1, 0.0, 0.0, 0.0, t0_2);

            let dp =
                du * t0 * v0.transpose() + u0 * dt * v0.transpose() + u0 * (t0 * dv.transpose());

            for k in 0..3 {
                for l in 0..3 {
                    dpdf[k][l][i][j] = dp[(k, l)];
                }
            }
        }
    }
    w0
}

pub fn wdwddw_invertible_fem(
    dw: &mut [[f64; 3]; 4],
    ddw: &mut [[[[f64; 3]; 3]; 4]; 4],
    pos: &[[f64; 3]; 4],
    pos_ref: &[[f64; 3]; 4],
    neohook: &dyn Fn(&mut [f64; 3], &mut [[f64; 3]; 3], f64, f64, f64) -> f64,
) -> f64 {
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

    let mut dfdu: [[f64; 3]; 4] = [[0.0; 3]; 4];
    diff_deformation_gradient_of_tet(
        &mut dfdu,
        &pos_ref[0],
        &pos_ref[1],
        &pos_ref[2],
        &pos_ref[3],
    );

    let (u0, s0, v0) = svd3::<f64>(f0, 30);

    let mut diff: [[[[f64; 3]; 3]; 3]; 9] = [[[[0.0; 3]; 3]; 3]; 9];
    svd3_differential(&mut diff, u0, s0, v0);

    let mut p0 = Mat3::zeros();
    let mut dpdf: [[[[f64; 3]; 3]; 3]; 3] = [[[[0.0; 3]; 3]; 3]; 3];

    let w0 = diff_piola_kirchhoff1st(&mut p0, &mut dpdf, u0, s0, v0, &diff, neohook);
    for ino in 0..4 {
        for i in 0..3 {
            dw[ino][i] =
                p0[(i, 0)] * dfdu[ino][0] + p0[(i, 1)] * dfdu[ino][1] + p0[(i, 2)] * dfdu[ino][2];
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
    w0
}
