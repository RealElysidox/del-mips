use nalgebra::Matrix3;
use nalgebra::Vector2;

type Vec2 = Vector2<f64>;
type Mat3 = Matrix3<f64>;

pub fn dwddw_beam2(
    dw: &mut [[f64; 3]; 2],
    ddw: &mut [[[[f64; 3]; 3]; 2]; 2],
    ei: f64,
    ae: f64,
    x: &[[f64; 2]; 2],
    u: &[[f64; 3]; 2],
) {
    let e_len = ((x[1][0] - x[0][0]).powi(2) + (x[1][1] - x[0][1]).powi(2)).sqrt();

    let mut ec: [[[[f64; 3]; 3]; 2]; 2] = [[[[0.0; 3]; 3]; 2]; 2];
    {
        let tmp1 = ei / (e_len * e_len * e_len);
        let tmp2 = ae / e_len;

        ec[0][0][0][0] = tmp2;
        ec[1][1][0][0] = tmp2;
        ec[1][0][0][0] = -tmp2;
        ec[0][1][0][0] = -tmp2;
        ec[0][0][1][1] = tmp1 * 12.0;
        ec[1][1][1][1] = tmp1 * 12.0;
        ec[1][0][1][1] = -tmp1 * 12.0;
        ec[0][1][1][1] = -tmp1 * 12.0;
        ec[0][0][1][2] = tmp1 * e_len * 6.0;
        ec[0][0][2][1] = tmp1 * e_len * 6.0;
        ec[1][0][2][1] = tmp1 * e_len * 6.0;
         ec[0][1][1][2] = tmp1 * e_len * 6.0;
        ec[0][1][2][1] = -tmp1 * e_len * 6.0;
       ec[1][0][1][2] = -tmp1 * e_len * 6.0;
        ec[1][1][1][2] = -tmp1 * e_len * 6.0;
        ec[1][1][2][1] = -tmp1 * e_len * 6.0;
         ec[0][0][2][2] = tmp1 * e_len * e_len * 4.0;
        ec[1][1][2][2] = tmp1 * e_len * e_len * 4.0;
         ec[1][0][2][2] = tmp1 * e_len * e_len * 2.0;
        ec[0][1][2][2] = tmp1 * e_len * e_len * 2.0;
    }

    let inv_e_len = 1.0 / e_len;
    let cs = [
        (x[1][0] - x[0][0]) * inv_e_len,
        (x[1][1] - x[0][1]) * inv_e_len,
    ];
    let e_r: [[f64; 3]; 3] = [
        [cs[0], -cs[1], 0.0],
        [cs[1], cs[0], 0.0],
        [0.0, 0.0, 1.0],
    ];

    for i in 0..2{
        for j in 0..2{
          for n in 0..3{
                for m in 0..3{
                     ddw[i][j][n][m] = 0.0;
                }
          }
        }
    }


    for i in 0..3 {
        for j in 0..3 {
            for n in 0..3 {
                for m in 0..3 {
                   ddw[0][0][i][j] += e_r[i][n] * ec[0][0][n][m] * e_r[j][m];
                    ddw[0][1][i][j] += e_r[i][n] * ec[0][1][n][m] * e_r[j][m];
                    ddw[1][0][i][j] += e_r[i][n] * ec[1][0][n][m] * e_r[j][m];
                   ddw[1][1][i][j] += e_r[i][n] * ec[1][1][n][m] * e_r[j][m];
                }
            }
        }
    }
    for n in 0..2 {
        for i in 0..3 {
          dw[n][i] = ddw[n][0][i][0] * u[0][0] + ddw[n][0][i][1] * u[0][1] + ddw[n][0][i][2] * u[0][2]
                + ddw[n][1][i][0] * u[1][0] + ddw[n][1][i][1] * u[1][1] + ddw[n][1][i][2] * u[1][2];
        }
    }
}