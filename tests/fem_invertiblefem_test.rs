#[cfg(test)]
mod tests {
    use del_mips::utils::*;
    use rand::{self, prelude::Distribution};

    #[test]
    fn unit_test() {
        // let mut rng = rand::thread_rng();
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut uni_dist = rand::distributions::Uniform::new(0.0, 1.0);
        const eps: f64 = 1.0e-5;

        for itr in 0..10000 {
            // random rest shape of a tetrahedron
            let pos0_cap = [random_vec(), random_vec(), random_vec(), random_vec()];
            let vol = volume_tet(pos0_cap[0], pos0_cap[1], pos0_cap[2], pos0_cap[3]);
            if vol < 0.01 {
                continue;
            }

            // random deformed shape of a tetrahedron
            let pos0 = [
                add_vec(pos0_cap[0], scale_vec(random_vec(), 0.2)),
                add_vec(pos0_cap[1], scale_vec(random_vec(), 0.2)),
                add_vec(pos0_cap[2], scale_vec(random_vec(), 0.2)),
                add_vec(pos0_cap[3], scale_vec(random_vec(), 0.2)),
            ];
        }

        // todo!();
    }
}
