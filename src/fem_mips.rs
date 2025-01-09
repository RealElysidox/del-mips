use nalgebra::Vector3;
use nalgebra::Matrix3;

type Vec3 = Vector3<f64>;
type Mat3 = Matrix3<f64>;

fn mat3_outer_product(a: &Vec3, b: &Vec3) -> Mat3 {
    Mat3::from_columns(&[
        a * b[0],
        a * b[1],
        a * b[2],
    ])
}

fn mat3_identity(val: f64) -> Mat3 {
    Mat3::new(
        val, 0.0, 0.0,
        0.0, val, 0.0,
        0.0, 0.0, val
    )
}

pub fn wdwddw_mips(
    energy: &mut f64,
    energy_gradient: &mut [[f64; 3]; 3],
    energy_hessian: &mut [[[[f64; 3]; 3]; 3]; 3],
    triangle_vertices: &[[f64; 3]; 3],
    reference_vertices: &[[f64; 3]; 3],
) {
    let p0 = Vec3::new(triangle_vertices[0][0], triangle_vertices[0][1], triangle_vertices[0][2]);
    let p1 = Vec3::new(triangle_vertices[1][0], triangle_vertices[1][1], triangle_vertices[1][2]);
    let p2 = Vec3::new(triangle_vertices[2][0], triangle_vertices[2][1], triangle_vertices[2][2]);
    let p_cap0 = Vec3::new(reference_vertices[0][0], reference_vertices[0][1], reference_vertices[0][2]);
    let p_cap1 = Vec3::new(reference_vertices[1][0], reference_vertices[1][1], reference_vertices[1][2]);
    let p_cap2 = Vec3::new(reference_vertices[2][0], reference_vertices[2][1], reference_vertices[2][2]);

    let v01 = p1 - p0;
    let v12 = p2 - p1;
    let v20 = p0 - p2;

    let n = v01.cross(&v20);
    let area = n.norm() * 0.5;
    let area_cap = ((p_cap1 - p_cap0).cross(&(p_cap2 - p_cap0))).norm() * 0.5;

    let la = (p1 - p2).norm_squared();
    let lb = (p2 - p0).norm_squared();
    let lc = (p0 - p1).norm_squared();
    let la_cap = (p_cap1 - p_cap2).norm_squared();
    let lb_cap = (p_cap2 - p_cap0).norm_squared();
    let lc_cap = (p_cap0 - p_cap1).norm_squared();

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

    let dec_d0 = t00 * p0 + t01 * p1 + t02 * p2;
    let dec_d1 = t01 * p0 + t11 * p1 + t12 * p2;
    let dec_d2 = t02 * p0 + t12 * p1 + t22 * p2;


    energy_gradient[0][0] = dec_d0.x;
    energy_gradient[0][1] = dec_d0.y;
    energy_gradient[0][2] = dec_d0.z;
    energy_gradient[1][0] = dec_d1.x;
    energy_gradient[1][1] = dec_d1.y;
    energy_gradient[1][2] = dec_d1.z;
    energy_gradient[2][0] = dec_d2.x;
    energy_gradient[2][1] = dec_d2.y;
    energy_gradient[2][2] = dec_d2.z;


    let tmp1 = 0.25 / area;
    let dad0 = ((v20.dot(&v12)) * v01 - (v01.dot(&v12)) * v20) * tmp1;
    let dad1 = ((v01.dot(&v20)) * v12 - (v12.dot(&v20)) * v01) * tmp1;
    let dad2 = ((v12.dot(&v01)) * v20 - (v20.dot(&v01)) * v12) * tmp1;


    let op = |a: &Vec3, b: &Vec3| mat3_outer_product(a, b);

    let ddad0d0 = (mat3_identity(v12.dot(&v12)) - op(&v12, &v12) - 4.0 * op(&dad0, &dad0)) * tmp1;
    let ddad0d1 = (mat3_identity(v20.dot(&v12)) - op(&v20, &(v12 - v01)) - op(&v01, &v20) - 4.0 * op(&dad0, &dad1)) * tmp1;
    let ddad0d2 = (mat3_identity(v01.dot(&v12)) - op(&v01, &(v12 - v20)) - op(&v20, &v01) - 4.0 * op(&dad0, &dad2)) * tmp1;
    let ddad1d0 = (mat3_identity(v12.dot(&v20)) - op(&v12, &(v20 - v01)) - op(&v01, &v12) - 4.0 * op(&dad1, &dad0)) * tmp1;
    let ddad1d1 = (mat3_identity(v20.dot(&v20)) - op(&v20, &v20) - 4.0 * op(&dad1, &dad1)) * tmp1;
    let ddad1d2 = (mat3_identity(v01.dot(&v20)) - op(&v01, &(v20 - v12)) - op(&v12, &v01) - 4.0 * op(&dad1, &dad2)) * tmp1;
    let ddad2d0 = (mat3_identity(v12.dot(&v01)) - op(&v12, &(v01 - v20)) - op(&v20, &v12) - 4.0 * op(&dad2, &dad0)) * tmp1;
    let ddad2d1 = (mat3_identity(v20.dot(&v01)) - op(&v20, &(v01 - v12)) - op(&v12, &v20) - 4.0 * op(&dad2, &dad1)) * tmp1;
    let ddad2d2 = (mat3_identity(v01.dot(&v01)) - op(&v01, &v01) - 4.0 * op(&dad2, &dad2)) * tmp1;

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

    let dd_ed0d0 = ec * dea * ddad0d0 + ec * ddea * op(&dad0, &dad0) + ea * mat3_identity(t00) + dea * op(&dad0, &dec_d0) * 2.0;
    let dd_ed0d1 = ec * dea * ddad0d1 + ec * ddea * op(&dad0, &dad1) + ea * mat3_identity(t01) + dea * op(&dad0, &dec_d1) + dea * op(&dad1, &dec_d0);
    let dd_ed0d2 = ec * dea * ddad0d2 + ec * ddea * op(&dad0, &dad2) + ea * mat3_identity(t02) + dea * op(&dad0, &dec_d2) + dea * op(&dad2, &dec_d0);
    let dd_ed1d0 = ec * dea * ddad1d0 + ec * ddea * op(&dad1, &dad0) + ea * mat3_identity(t01) + dea * op(&dad1, &dec_d0) + dea * op(&dad0, &dec_d1);
    let dd_ed1d1 = ec * dea * ddad1d1 + ec * ddea * op(&dad1, &dad1) + ea * mat3_identity(t11) + dea * op(&dad1, &dec_d1) * 2.0;
    let dd_ed1d2 = ec * dea * ddad1d2 + ec * ddea * op(&dad1, &dad2) + ea * mat3_identity(t12) + dea * op(&dad1, &dec_d2) + dea * op(&dad2, &dec_d1);
    let dd_ed2d0 = ec * dea * ddad2d0 + ec * ddea * op(&dad2, &dad0) + ea * mat3_identity(t02) + dea * op(&dad2, &dec_d0) + dea * op(&dad0, &dec_d2);
    let dd_ed2d1 = ec * dea * ddad2d1 + ec * ddea * op(&dad2, &dad1) + ea * mat3_identity(t12) + dea * op(&dad2, &dec_d1) + dea * op(&dad1, &dec_d2);
    let dd_ed2d2 = ec * dea * ddad2d2 + ec * ddea * op(&dad2, &dad2) + ea * mat3_identity(t22) + dea * op(&dad2, &dec_d2) * 2.0;


    for idim in 0..3 {
        for jdim in 0..3 {
            energy_hessian[0][0][idim][jdim] = dd_ed0d0[(idim, jdim)];
            energy_hessian[0][1][idim][jdim] = dd_ed0d1[(idim, jdim)];
            energy_hessian[0][2][idim][jdim] = dd_ed0d2[(idim, jdim)];
            energy_hessian[1][0][idim][jdim] = dd_ed1d0[(idim, jdim)];
            energy_hessian[1][1][idim][jdim] = dd_ed1d1[(idim, jdim)];
            energy_hessian[1][2][idim][jdim] = dd_ed1d2[(idim, jdim)];
            energy_hessian[2][0][idim][jdim] = dd_ed2d0[(idim, jdim)];
            energy_hessian[2][1][idim][jdim] = dd_ed2d1[(idim, jdim)];
            energy_hessian[2][2][idim][jdim] = dd_ed2d2[(idim, jdim)];
        }
    }
}