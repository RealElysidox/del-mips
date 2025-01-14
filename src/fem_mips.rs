use del_geo_core::mat3_col_major::from_diagonal;
use del_geo_core::mat3_row_major::*;
use del_geo_core::vec3::Vec3;
use del_geo_core::vec3::*;

fn from_outer_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 9] {
    [
        a[0] * b[0],
        a[0] * b[1],
        a[0] * b[2],
        a[1] * b[0],
        a[1] * b[1],
        a[1] * b[2],
        a[2] * b[0],
        a[2] * b[1],
        a[2] * b[2],
    ]
}

pub fn wdwddw_mips(
    energy: &mut f64,
    energy_gradient: &mut [[f64; 3]; 3],
    energy_hessian: &mut [[[[f64; 3]; 3]; 3]; 3],
    triangle_vertices: &[[f64; 3]; 3],
    reference_vertices: &[[f64; 3]; 3],
) {
    let p0 = triangle_vertices[0];
    let p1 = triangle_vertices[1];
    let p2 = triangle_vertices[2];
    let p_cap0 = reference_vertices[0];
    let p_cap1 = reference_vertices[1];
    let p_cap2 = reference_vertices[2];

    let v01 = p1.sub(&p0);
    let v12 = p2.sub(&p1);
    let v20 = p0.sub(&p2);

    let n = cross(&v01, &v20);
    let area = norm(&n) * 0.5;
    let area_cap = p_cap1.sub(&p_cap0).cross(&p_cap2.sub(&p_cap0)).norm() * 0.5;

    let la = v12.squared_norm();
    let lb = v20.squared_norm();
    let lc = v01.squared_norm();
    let la_cap = p_cap1.sub(&p_cap2).squared_norm();
    let lb_cap = p_cap2.sub(&p_cap0).squared_norm();
    let lc_cap = p_cap0.sub(&p_cap1).squared_norm();

    let cot0 = -la + lb + lc;
    let cot1 = la - lb + lc;
    let cot2 = la + lb - lc;

    let tmp0 = 1.0 / (8.0 * area_cap);

    let ec = (cot0 * la_cap + cot1 * lb_cap + cot2 * lc_cap) * tmp0;

    let t00 = 4.0 * la_cap * tmp0;
    let t11 = 4.0 * lb_cap * tmp0;
    let t22 = 4.0 * lc_cap * tmp0;
    let t01 = (-2.0 * la_cap - 2.0 * lb_cap + 2.0 * lc_cap) * tmp0;
    let t02 = (-2.0 * la_cap + 2.0 * lb_cap - 2.0 * lc_cap) * tmp0;
    let t12 = (2.0 * la_cap - 2.0 * lb_cap - 2.0 * lc_cap) * tmp0;

    let dec_d0 = p0.scale(t00).add(&p1.scale(t01)).add(&p2.scale(t02));
    let dec_d1 = p0.scale(t01).add(&p1.scale(t11)).add(&p2.scale(t12));
    let dec_d2 = p0.scale(t02).add(&p1.scale(t12)).add(&p2.scale(t22));

    energy_gradient[0][0] = dec_d0[0];
    energy_gradient[0][1] = dec_d0[1];
    energy_gradient[0][2] = dec_d0[2];
    energy_gradient[1][0] = dec_d1[0];
    energy_gradient[1][1] = dec_d1[1];
    energy_gradient[1][2] = dec_d1[2];
    energy_gradient[2][0] = dec_d2[0];
    energy_gradient[2][1] = dec_d2[1];
    energy_gradient[2][2] = dec_d2[2];

    let tmp1 = 0.25 / area;
    let dad0 = v01
        .scale(v20.dot(&v12))
        .sub(&v20.scale(v01.dot(&v12)))
        .scale(tmp1);
    let dad1 = v12
        .scale(v01.dot(&v20))
        .sub(&v01.scale(v12.dot(&v20)))
        .scale(tmp1);
    let dad2 = v20
        .scale(v12.dot(&v01))
        .sub(&v12.scale(v20.dot(&v01)))
        .scale(tmp1);

    let op = |a: &[f64; 3], b: &[f64; 3]| from_outer_product(a, b);

    let ddad0d0 = from_diagonal(&[v12.dot(&v12); 3]).sub(&op(&v12, &v12)).sub(&op(&dad0, &dad0).scale(4.0)).scale(tmp1);
    let ddad0d1 = from_diagonal(&[v20.dot(&v12); 3]).sub(&op(&v20, &v12.sub(&v01))).sub(&op(&v01, &v20)).sub(&op(&dad0, &dad1).scale(4.0)).scale(tmp1);
    let ddad0d2 = from_diagonal(&[v01.dot(&v12); 3]).sub(&op(&v01, &v12.sub(&v20))).sub(&op(&v20, &v01)).sub(&op(&dad0, &dad2).scale(4.0)).scale(tmp1);
    let ddad1d0 = from_diagonal(&[v12.dot(&v20); 3]).sub(&op(&v12, &v20.sub(&v01))).sub(&op(&v01, &v12)).sub(&op(&dad1, &dad0).scale(4.0)).scale(tmp1);
    let ddad1d1 = from_diagonal(&[v20.dot(&v20); 3]).sub(&op(&v20, &v20)).sub(&op(&dad1, &dad1).scale(4.0)).scale(tmp1);
    let ddad1d2 = from_diagonal(&[v01.dot(&v20); 3]).sub(&op(&v01, &v20.sub(&v12))).sub(&op(&v12, &v01)).sub(&op(&dad1, &dad2).scale(4.0)).scale(tmp1);
    let ddad2d0 = from_diagonal(&[v12.dot(&v01); 3]).sub(&op(&v12, &v01.sub(&v20))).sub(&op(&v20, &v12)).sub(&op(&dad2, &dad0).scale(4.0)).scale(tmp1);
    let ddad2d1 = from_diagonal(&[v20.dot(&v01); 3]).sub(&op(&v20, &v01.sub(&v12))).sub(&op(&v12, &v20)).sub(&op(&dad2, &dad1).scale(4.0)).scale(tmp1);
    let ddad2d2 = from_diagonal(&[v01.dot(&v01); 3]).sub(&op(&v01, &v01)).sub(&op(&dad2, &dad2).scale(4.0)).scale(tmp1);
    
    let adr = area_cap / area + area / area_cap;
    let ea = adr;
    let dadr = 1.0 / area_cap - area_cap / (area * area);
    let dea = dadr;
    let ddadr = 2.0 * area_cap / (area * area * area);
    let ddea = ddadr;

    *energy = ec * ea;

    for idim in 0..3 {
        energy_gradient[0][idim] = ec * dea * dad0[idim] + ea * dec_d0[idim];
        energy_gradient[1][idim] = ec * dea * dad1[idim] + ea * dec_d1[idim];
        energy_gradient[2][idim] = ec * dea * dad2[idim] + ea * dec_d2[idim];
    }

    let dded0d0 = &ddad0d0.scale(ec * dea).add(&op(&dad0, &dad0).scale(ec * ddea)).add(&(from_diagonal(&[t00; 3]).scale(ea))).add(&(op(&dad0, &dec_d0).scale(dea * 2.0)));
    let dded0d1 = &ddad0d1.scale(ec * dea).add(&op(&dad0, &dad1).scale(ec * ddea)).add(&(from_diagonal(&[t01; 3]).scale(ea))).add(&(op(&dad0, &dec_d1).scale(dea))).add(&(op(&dad1, &dec_d0).scale(dea)));
    let dded0d2 = &ddad0d2.scale(ec * dea).add(&op(&dad0, &dad2).scale(ec * ddea)).add(&(from_diagonal(&[t02; 3]).scale(ea))).add(&(op(&dad0, &dec_d2).scale(dea))).add(&(op(&dad2, &dec_d0).scale(dea)));
    let dded1d0 = &ddad1d0.scale(ec * dea).add(&op(&dad1, &dad0).scale(ec * ddea)).add(&(from_diagonal(&[t01; 3]).scale(ea))).add(&(op(&dad1, &dec_d0).scale(dea))).add(&(op(&dad0, &dec_d1).scale(dea)));
    let dded1d1 = &ddad1d1.scale(ec * dea).add(&op(&dad1, &dad1).scale(ec * ddea)).add(&(from_diagonal(&[t11; 3]).scale(ea))).add(&(op(&dad1, &dec_d1).scale(dea * 2.0)));
    let dded1d2 = &ddad1d2.scale(ec * dea).add(&op(&dad1, &dad2).scale(ec * ddea)).add(&(from_diagonal(&[t12; 3]).scale(ea))).add(&(op(&dad1, &dec_d2).scale(dea))).add(&(op(&dad2, &dec_d1).scale(dea)));
    let dded2d0 = &ddad2d0.scale(ec * dea).add(&op(&dad2, &dad0).scale(ec * ddea)).add(&(from_diagonal(&[t02; 3]).scale(ea))).add(&(op(&dad2, &dec_d0).scale(dea))).add(&(op(&dad0, &dec_d2).scale(dea)));
    let dded2d1 = &ddad2d1.scale(ec * dea).add(&op(&dad2, &dad1).scale(ec * ddea)).add(&(from_diagonal(&[t12; 3]).scale(ea))).add(&(op(&dad2, &dec_d1).scale(dea))).add(&(op(&dad1, &dec_d2).scale(dea)));
    let dded2d2 = &ddad2d2.scale(ec * dea).add(&op(&dad2, &dad2).scale(ec * ddea)).add(&(from_diagonal(&[t22; 3]).scale(ea))).add(&(op(&dad2, &dec_d2).scale(dea * 2.0)));

    for idim in 0..3 {
        for jdim in 0..3 {
            energy_hessian[0][0][idim][jdim] = dded0d0[idim * 3 + jdim];
            energy_hessian[0][1][idim][jdim] = dded0d1[idim * 3 + jdim];
            energy_hessian[0][2][idim][jdim] = dded0d2[idim * 3 + jdim];
            energy_hessian[1][0][idim][jdim] = dded1d0[idim * 3 + jdim];
            energy_hessian[1][1][idim][jdim] = dded1d1[idim * 3 + jdim];
            energy_hessian[1][2][idim][jdim] = dded1d2[idim * 3 + jdim];
            energy_hessian[2][0][idim][jdim] = dded2d0[idim * 3 + jdim];
            energy_hessian[2][1][idim][jdim] = dded2d1[idim * 3 + jdim];
            energy_hessian[2][2][idim][jdim] = dded2d2[idim * 3 + jdim];
        }
    }
}
