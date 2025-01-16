#[allow(unused_variables)]
#[cfg(test)]
mod tests {
    use del_geo_core::mat3_row_major::*;
    use del_geo_core::{mat3_col_major::from_diagonal, mat3_row_major::svd};
    use del_mips::fem_invertiblefem::*;
    use del_mips::{
        geo_tet::{deformation_gradient_of_tet, diff_deformation_gradient_of_tet},
        utils::*,
    };

    #[test]
    fn unit_test_df() {
        // let mut rng = rand::thread_rng();
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1926);
        let mut uni_dist = rand::distributions::Uniform::new(0.0, 1.0);
        const EPS: f64 = 1.0e-5;

        // Test for dF
        for _ in 0..10000 {
            // random rest shape of a tetrahedron
            let pos0_cap = [random_vec(), random_vec(), random_vec(), random_vec()];
            let vol = volume_tet(pos0_cap[0], pos0_cap[1], pos0_cap[2], pos0_cap[3]);
            if vol < 0.01 {
                continue;
            }

            // random deformed shape of a tetrahedron
            let pos0 = [
                add_vec(&pos0_cap[0], &scale_vec(&random_vec(), 0.2)),
                add_vec(&pos0_cap[1], &scale_vec(&random_vec(), 0.2)),
                add_vec(&pos0_cap[2], &scale_vec(&random_vec(), 0.2)),
                add_vec(&pos0_cap[3], &scale_vec(&random_vec(), 0.2)),
            ];

            let f0 = deformation_gradient_of_tet(
                &pos0_cap[0],
                &pos0_cap[1],
                &pos0_cap[2],
                &pos0_cap[3],
                &pos0[0],
                &pos0[1],
                &pos0[2],
                &pos0[3],
            );
            let dfdu = diff_deformation_gradient_of_tet(
                &pos0_cap[0],
                &pos0_cap[1],
                &pos0_cap[2],
                &pos0_cap[3],
            );
            let basis0 = mat3_from_3basis(
                &sub_vec(&pos0_cap[1], &pos0_cap[0]),
                &sub_vec(&pos0_cap[2], &pos0_cap[0]),
                &sub_vec(&pos0_cap[3], &pos0_cap[0]),
            );

            // Check dfdu
            for ino in 0..4 {
                for idim in 0..3 {
                    let mut a = [[0.0; 3]; 3];
                    random_2d_matrix(&mut a, &mut uni_dist, &mut rng, 1.0, 0.0);
                    let mut pos1 = pos0.clone();
                    pos1[ino][idim] += EPS;
                    let basis1 = mat3_from_3basis(
                        &sub_vec(&pos1[1], &pos1[0]),
                        &sub_vec(&pos1[2], &pos1[0]),
                        &sub_vec(&pos1[3], &pos1[0]),
                    );
                    let f1 = mul_mat3x3(&basis1, &inverse_mat3(&basis0));
                    let v0 =
                        trace_mat3x3(&mul_mat3x3(&transpose_mat3x3(&a), &(sub_mat3x3(&f1, &f0))))
                            / EPS;
                    let v1 = a[idim][0] * dfdu[ino][0]
                        + a[idim][1] * dfdu[ino][1]
                        + a[idim][2] * dfdu[ino][2];
                    assert!((v0 - v1).abs() < 1e-3);
                }
            }
        }
    }

    fn neohook(l0: f64, l1: f64, l2: f64) -> (f64, [f64; 3], [[f64; 3]; 3]) {
        let dw = [l0 - 1.0, l1 - 1.0, l2 - 1.0];
        let ddw = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let energy = (l0 - 1.0).powi(2) * 0.5 + (l1 - 1.0).powi(2) * 0.5 + (l2 - 1.0).powi(2) * 0.5;
        (energy, dw, ddw)
    }

    #[test]
    fn unit_test_dpdf() {
        const EPS: f64 = 1.0e-5;
        for _ in 0..10000 {
            // random rest shape of a tetrahedron
            let pos0_cap = [random_vec(), random_vec(), random_vec(), random_vec()];
            let vol = volume_tet(pos0_cap[0], pos0_cap[1], pos0_cap[2], pos0_cap[3]);
            if vol < 0.01 {
                continue;
            }

            // random deformed shape of a tetrahedron
            let pos0 = [
                add_vec(&pos0_cap[0], &scale_vec(&random_vec(), 0.2)),
                add_vec(&pos0_cap[1], &scale_vec(&random_vec(), 0.2)),
                add_vec(&pos0_cap[2], &scale_vec(&random_vec(), 0.2)),
                add_vec(&pos0_cap[3], &scale_vec(&random_vec(), 0.2)),
            ];

            let f0 = deformation_gradient_of_tet(
                &pos0_cap[0],
                &pos0_cap[1],
                &pos0_cap[2],
                &pos0_cap[3],
                &pos0[0],
                &pos0[1],
                &pos0[2],
                &pos0[3],
            );

            let f0_flat: [f64; 9] = [
                f0[0][0], f0[0][1], f0[0][2], f0[1][0], f0[1][1], f0[1][2], f0[2][0], f0[2][1],
                f0[2][2],
            ];
            let (u0, g0, v0) = svd(&f0_flat, 30);
            assert!(
                u0.mult_mat_row_major(&g0)
                    .mult_mat_row_major(&v0)
                    .squared_norm()
                    < 1.0e-20
            );
            let diff = svd_differential(u0, g0, v0);
            let (w0, p0, dpdf) = diff_piola_kirchhoff1st(&u0, &g0, &v0, &diff, &neohook);
            let (wt, dwdl0, ddwddl) = neohook(g0[0], g0[1], g0[2]);

            // Check dpdf
            for i in 0..3 {
                for j in 0..3 {
                    let mut f1 = f0.clone();
                    f1[i][j] += EPS;
                    let f1_flat: [f64; 9] = [
                        f1[0][0], f1[0][1], f1[0][2], f1[1][0], f1[1][1], f1[1][2], f1[2][0],
                        f1[2][1], f1[2][2],
                    ];
                    let (u1, s1, v1) = svd(&f1_flat, 30);
                    let (w1, dwdl1, ddwddl1) = neohook(s1[0], s1[4], s1[8]);
                    assert!(((w1 - w0) / EPS - p0[3 * i + j]).abs() < 1.0e-3);
                    {
                        let dwdf = dwdl0[0] * diff[3][i][j]
                            + dwdl0[1] * diff[4][i][j]
                            + dwdl0[2] * diff[5][i][j];
                        assert!((dwdf - p0[3 * i + j]).abs() < 1.0e-8);
                    }
                    let t1 = from_diagonal(&[dwdl1[0], dwdl1[1], dwdl1[2]]);
                    let p1 = u1.mult_mat_row_major(&t1.mult_mat_row_major(&transpose(&v1)));
                    for k in 0..3 {
                        for l in 0..3 {
                            let v0 = p1.sub(&p0)[3 * k + l] / EPS;
                            assert!((v0 - dpdf[k][l][i][j]).abs() < 2.0e-5);
                            assert!((dpdf[k][l][i][j] - dpdf[i][j][k][l]).abs() < 1.0e-5);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unit_test_energy() {
        const EPS: f64 = 1.0e-5;
        for _ in 0..10000 {
            // make random rest shape of a tetrahedron
            let pos0_cap = [random_vec(), random_vec(), random_vec(), random_vec()];
            let vol = volume_tet(pos0_cap[0], pos0_cap[1], pos0_cap[2], pos0_cap[3]);
            if vol < 0.01 {
                continue;
            }

            // make random deformed shape of a tetrahedron
            let pos0 = [random_vec(), random_vec(), random_vec(), random_vec()];

            let (w0, dw0, ddw0) = wdwddw_invertible_fem(&pos0_cap, &pos0, &neohook);

            for ino in 0..4 {
                for jno in 0..4 {
                    for idim in 0..3 {
                        for jdim in 0..3 {
                            let val0 = ddw0[ino][jno][idim][jdim];
                            let val1 = ddw0[jno][ino][jdim][idim];
                            assert!((val0 - val1).abs() < 1.0e-5);
                        }
                    }
                }
            }

            for ino in 0..4 {
                for idim in 0..3 {
                    let mut pos1 = pos0.clone();
                    pos1[ino][idim] += EPS;
                    let (w1, dw1, ddw1) = wdwddw_invertible_fem(&pos0_cap, &pos1, &neohook);
                    {
                        let v0 = (w1 - w0) / EPS;
                        let v1 = dw0[ino][idim];
                        assert!((v0 - v1).abs() < 2.0e-3 * (1.0 + v1.abs()));
                    }
                    for jno in 0..4 {
                        for jdim in 0..3 {
                            let v0 = (dw1[jno][jdim] - dw0[jno][jdim]) / EPS;
                            let v1 = ddw0[jno][ino][jdim][idim];
                            assert!((v0 - v1).abs() < 2.0e-3 * (1.0 + v1.abs()));
                        }
                    }
                }
            }
        }
    }
}
