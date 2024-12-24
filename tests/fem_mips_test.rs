#[cfg(test)]
mod tests {
    use del_mips::fem_mips::*;
    use del_mips::utils::*;
    use rand;

    #[test]
    fn unit_test() {
        let mut rng = rand::thread_rng();
        let mut uni_dist = rand::distributions::Uniform::new(0.0, 1.0);
        const eps: f64 = 1.0e-3;

        for itr in 0..10000 {
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
            let p = mat3_rot_from_axis_angle_vec(&[0.3, 1.0, 0.5]);
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
                    c[ino][idim] += eps;
                    let mut e1 = 0.0;
                    let mut de1 = [[0.0; 3]; 3];
                    let mut dde1 = [[[[0.0; 3]; 3]; 3]; 3];
                    wdwddw_mips(&mut e1, &mut de1, &mut dde1, &c, &P);
                    {
                        let val0 = (e1 - e) / eps;
                        let val1 = de[ino][idim];
                        assert!((val0 - val1).abs() < 1.0e-2 * (1.0 + val1.abs()));
                    }
                    for jno in 0..3 {
                        for jdim in 0..3 {
                            let val0 = (de1[jno][jdim] - de[jno][jdim]) / eps;
                            let val1 = dde[jno][ino][jdim][idim];
                            dbg!(val0, val1);
                            assert!((val0 - val1).abs() < 3.0e-2 * (1.0 + val1.abs()));
                        }
                    }
                }
            }
        }
    }
}