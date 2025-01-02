#[cfg(test)]
mod tests {
    use del_mips::utils::*;
    use rand::{self, prelude::Distribution};

    #[test]
    fn unit_test() {
        let mut rng = rand::thread_rng();
        let mut uni_dist = rand::distributions::Uniform::new(0.0, 1.0);
        const eps: f64 = 1.0e-5;

        for itr in 0..10000 {
            let pos0 = [random_vec(), random_vec(), random_vec(), random_vec()];
            let vol = volume_tet(pos0[0], pos0[1], pos0[2], pos0[3]);
            if vol < 0.01 {
                continue;
            }
        }

        // todo!();
    }
}
