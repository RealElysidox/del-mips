#![allow(dead_code)]
use del_geo_core::mat3_row_major::*;

fn sqr(x: f64) -> f64 {
    x * x
}

fn energy_s(a: f64, b: f64, c: f64, m: &str) -> f64 {
    match m {
        "ARAP" => sqr(a - 1.0) + sqr(b - 1.0) + sqr(c - 1.0),
        "SymDirichlet" => sqr(a) + 1.0 / sqr(a) + sqr(b) + 1.0 / sqr(b) + sqr(c) + 1.0 / sqr(c),
        "MIPS" => (sqr(a) + sqr(b) + sqr(c)) / (a * b * c),
        "Ogden" => (0..5)
            .map(|k| a.powf(0.5f64.powi(k)) + b.powf(0.5f64.powi(k)) + c.powf(0.5f64.powi(k)) - 3.0)
            .sum(),
        "Yeoh" => (0..3)
            .map(|k| (sqr(a) + sqr(b) + sqr(c) - 3.0).powi(k + 1))
            .sum(),
        _ => 0.0,
    }
}

fn grad_energy_s(a: f64, b: f64, c: f64, m: &str) -> [f64; 3] {
    let yeoh_coeff: f64 = (0..3)
        .map(|i| 2.0 * (i + 1) as f64 * (sqr(a) + sqr(b) + sqr(c) - 3.0).powi(i))
        .sum();
    match m {
        "ARAP" => [2.0 * (a - 1.0), 2.0 * (b - 1.0), 2.0 * (c - 1.0)],
        "SymDirichlet" => [
            2.0 - 2.0 / a.powi(3),
            2.0 - 2.0 / b.powi(3),
            2.0 - 2.0 / c.powi(3),
        ],
        "MIPS" => [
            1.0 / (b * c) - (b / c + c / b) / sqr(a),
            1.0 / (a * c) - (a / c + c / a) / sqr(b),
            1.0 / (a * b) - (a / b + b / a) / sqr(c),
        ],
        "Ogden" => [
            (0..5)
                .map(|i| a.powf(0.5f64.powi(i) - 1.0) / 2.0f64.powi(i))
                .sum(),
            (0..5)
                .map(|i| b.powf(0.5f64.powi(i) - 1.0) / 2.0f64.powi(i))
                .sum(),
            (0..5)
                .map(|i| c.powf(0.5f64.powi(i) - 4.0) * (1.0 / 2.0f64.powi(i) - 3.0))
                .sum(),
        ],
        "Yeoh" => [yeoh_coeff * a, yeoh_coeff * b, yeoh_coeff * c],
        _ => [0.0, 0.0, 0.0],
    }
}

fn hessian_energy_s(a: f64, b: f64, c: f64, model: &str) -> [f64; 9] {
    let yeoh_coeff_i: f64 = (0..3)
        .map(|i| 2.0 * (i + 1) as f64 * (sqr(a) + sqr(b) + sqr(c) - 3.0).powi(i))
        .sum();
    let yeoh_coeff_rr = (0..2)
        .map(|i| 4.0 * (i + 1) as f64 * (i + 2) as f64 * (sqr(a) + sqr(b) + sqr(c) - 3.0).powi(i)).sum();
    let rr = [a * a, a * b, a* c, b * a, b * b, b * c, c * a, c * b, c * c];
    // row major order
    match model {
        "ARAP" => [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
        "SymDirichlet" => [
            2.0 + 6.0 / a.powi(4),
            0.0,
            0.0,
            0.0,
            2.0 + 6.0 / b.powi(4),
            0.0,
            0.0,
            0.0,
            2.0 + 6.0 / c.powi(4),
        ],
        "MIPS" => [
            (b / c + c / b) * 2.0 / a.powi(3),
            c / (sqr(a) * sqr(b)) - 1.0 / sqr(a) / c - 1.0 / sqr(b) / c,
            b / (sqr(a) * sqr(c)) - 1.0 / sqr(a) / b - 1.0 / sqr(c) / b,
            c / (sqr(a) * sqr(b)) - 1.0 / sqr(a) / c - 1.0 / sqr(b) / c,
            (a / c + c / a) * 2.0 / b.powi(3),
            a / (sqr(b) * sqr(c)) - 1.0 / sqr(c) / a - 1.0 / sqr(b) / a,
            b / (sqr(a) * sqr(c)) - 1.0 / sqr(a) / b - 1.0 / sqr(c) / b,
            a / (sqr(b) * sqr(c)) - 1.0 / sqr(c) / a - 1.0 / sqr(b) / a,
            (a / b + b / a) * 2.0 / c.powi(3),
        ],
        "Ogden" => [
            (0..5)
                .map(|i| {
                    a.powf(0.5f64.powi(i) - 2.0) / 2.0f64.powi(i) * (1.0 / 2.0f64.powi(i) - 1.0)
                })
                .sum(),
            0.0,
            0.0,
            0.0,
            (0..5)
                .map(|i| {
                    b.powf(0.5f64.powi(i) - 2.0) / 2.0f64.powi(i) * (1.0 / 2.0f64.powi(i) - 1.0)
                })
                .sum(),
            0.0,
            0.0,
            0.0,
            (0..5)
                .map(|i| {
                    c.powf(0.5f64.powi(i) - 5.0)
                        * (1.0 / 2.0f64.powi(i) - 3.0)
                        * (1.0 / 2.0f64.powi(i) - 4.0)
                })
                .sum(),
        ],
        "Yeoh" => rr.scale(yeoh_coeff_rr).add(&from_identity().scale(yeoh_coeff_i)),
        _ => [0.0; 9],
    }
}
