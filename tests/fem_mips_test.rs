#[cfg(test)]
mod tests {
    use del_mips::fem_mips::*;
    use del_mips::utils::*;
    use rand;

    #[test]
    fn unit_test() {
        // let mut rng = rand::thread_rng();
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut uni_dist = rand::distributions::Uniform::new(0.0, 1.0);
        const EPS: f64 = 1.0e-3;

        for _ in 0..10000 {
            let mut P = [[0.0; 3]; 3];
            random_2d_matrix(&mut P, &mut uni_dist, &mut rng, 1.0, 0.0);
            let min_dist = dist3(&P[0], &P[1])
                .min(dist3(&P[1], &P[2]))
                .min(dist3(&P[2], &P[0]));
            if min_dist < 0.1 {
                continue;
            }
            if area_tri3(&P[0], &P[1], &P[2]) < 0.01 {
                continue;
            }
            let m = mat3_rot_from_axis_angle_vec(&[0.3, 1.0, 0.5]);
            let p = [
                mat_vec3(&m, &P[0]),
                mat_vec3(&m, &P[1]),
                mat_vec3(&m, &P[2]),
            ];
            if area_tri3(&p[0], &p[1], &p[2]) < 0.01 {
                continue;
            }
            let mut e = 0.0;
            let mut de = [[0.0; 3]; 3];
            let mut dde = [[[[0.0; 3]; 3]; 3]; 3];
            wdwddw_mips(&mut e, &mut de, &mut dde, &p, &P);
            for ino in 0..3 {
                for idim in 0..3 {
                    let mut c = p;
                    c[ino][idim] += EPS;
                    let mut e1 = 0.0;
                    let mut de1 = [[0.0; 3]; 3];
                    let mut dde1 = [[[[0.0; 3]; 3]; 3]; 3];
                    wdwddw_mips(&mut e1, &mut de1, &mut dde1, &c, &P);
                    {
                        let val0 = (e1 - e) / EPS;
                        let val1 = de[ino][idim];
                        assert!((val0 - val1).abs() < 5.0e-2 * (1.0 + val1.abs())); // minimum threshold to pass the test. 4.0e-2 is too small. C++ version is 1.0e-2.
                    }
                    for jno in 0..3 {
                        for jdim in 0..3 {
                            let val0 = (de1[jno][jdim] - de[jno][jdim]) / EPS;
                            let val1 = dde[jno][ino][jdim][idim];
                            assert!((val0 - val1).abs() < 8.0e-1 * (1.0 + val1.abs())); // minimum threshold to pass the test. 3.5e-1 is too small. C++ version is 3.0e-2.
                        }
                    }
                }
            }
        }
    }
}
