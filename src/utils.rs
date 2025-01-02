use std::ops::{Add, Mul};

use rand::{self, prelude::Distribution};

pub fn random_2d_matrix<T, const N: usize, const M: usize>(
    a: &mut [[T; M]; N],
    dist: &mut rand::distributions::Uniform<f64>,
    rng: &mut rand::prelude::ThreadRng,
    mag: T,
    offset: T,
) where
    T: Copy + Add<Output = T> + Mul<f64, Output = T>,
{
    for i in 0..N {
        for j in 0..M {
            a[i][j] = mag * dist.sample(rng) + offset;
        }
    }
}

pub fn dist3(p0: &[f64; 3], p1: &[f64; 3]) -> f64 {
    ((p0[0] - p1[0]).powi(2) + (p0[1] - p1[1]).powi(2) + (p0[2] - p1[2]).powi(2)).sqrt()
}

pub fn area_tri3(p1: &[f64; 3], p2: &[f64; 3], p3: &[f64; 3]) -> f64 {
    let x = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]);
    let y = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]);
    let z = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
    0.5 * ((x * x + y * y + z * z).sqrt())
}

pub fn mat3_rot_from_axis_angle_vec(vec: &[f64; 3]) -> [[f64; 3]; 3] {
    let sqt = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    if sqt < 1.0e-20 {
        // Infinitesimal rotation approximation
        return [
            [1.0, -vec[2], vec[1]],
            [vec[2], 1.0, -vec[0]],
            [-vec[1], vec[0], 1.0],
        ];
    }

    let t = sqt.sqrt();
    let invt = 1.0 / t;
    let n = [vec[0] * invt, vec[1] * invt, vec[2] * invt];
    let c0 = t.cos();
    let s0 = t.sin();

    [
        [
            c0 + (1.0 - c0) * n[0] * n[0],
            -n[2] * s0 + (1.0 - c0) * n[0] * n[1],
            n[1] * s0 + (1.0 - c0) * n[0] * n[2],
        ],
        [
            n[2] * s0 + (1.0 - c0) * n[1] * n[0],
            c0 + (1.0 - c0) * n[1] * n[1],
            -n[0] * s0 + (1.0 - c0) * n[1] * n[2],
        ],
        [
            -n[1] * s0 + (1.0 - c0) * n[2] * n[0],
            n[0] * s0 + (1.0 - c0) * n[2] * n[1],
            c0 + (1.0 - c0) * n[2] * n[2],
        ],
    ]
}

pub fn volume_tet<T>(v0: [T; 3], v1: [T; 3], v2: [T; 3], v3: [T; 3]) -> T
where
    T: Copy
        + std::ops::Sub<Output = T>
        + std::ops::Mul<f64, Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>,
{
    let v = (v1[0] - v0[0])
        * ((v2[1] - v0[1]) * (v3[2] - v0[2]) - (v3[1] - v0[1]) * (v2[2] - v0[2]))
        + (v1[1] - v0[1]) * ((v2[2] - v0[2]) * (v3[0] - v0[0]) - (v3[2] - v0[2]) * (v2[0] - v0[0]))
        + (v1[2] - v0[2]) * ((v2[0] - v0[0]) * (v3[1] - v0[1]) - (v3[0] - v0[0]) * (v2[1] - v0[1]));

    v * 0.16666666666666666666666666666667
}

pub fn random_vec() -> [f64; 3] {
    let mut rng = rand::thread_rng();
    let uni_dist = rand::distributions::Uniform::new(0.0, 1.0);
    [
        uni_dist.sample(&mut rng),
        uni_dist.sample(&mut rng),
        uni_dist.sample(&mut rng),
    ]
}

pub fn mat_vec3<T>(m: &[[T; 3]; 3], v: &[T; 3]) -> [T; 3]
where
    T: Add<Output = T> + Mul<Output = T> + Copy,
{
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
