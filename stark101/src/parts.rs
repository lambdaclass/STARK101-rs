use crate::{
    channel::Channel, field::FieldElement, merkle_tree::MerkleTree, polynomial::Polynomial,
};

pub fn part1() -> (
    Vec<FieldElement>,
    FieldElement,
    Vec<FieldElement>,
    FieldElement,
    Vec<FieldElement>,
    Vec<FieldElement>,
    Polynomial,
    Vec<FieldElement>,
    MerkleTree,
    Channel,
) {
    let mut t = vec![FieldElement::one(), FieldElement::new(3141592)];
    let mut n = 2usize;
    while t.len() < 1023 {
        t.push(t[n - 2] * t[n - 2] + t[n - 1] * t[n - 1]);
        n += 1;
    }
    let g = FieldElement::generator().pow(3 * 2usize.pow(20));
    let points: Vec<FieldElement> = (0..1024).into_iter().map(|i| g.pow(i)).collect();
    let w = FieldElement::generator();
    let exp = (2usize.pow(30) * 3) / 8192;
    let h_gen = w.pow(exp);
    let h: Vec<FieldElement> = (0..8192).into_iter().map(|i| h_gen.pow(i)).collect();
    let domain: Vec<FieldElement> = h.clone().into_iter().map(|x| w * x).collect();
    let x_values: Vec<FieldElement> = points.clone().into_iter().rev().skip(1).rev().collect();
    let p: Polynomial = Polynomial::interpolate(&x_values, &t);
    let ev: Vec<FieldElement> = domain
        .clone()
        .into_iter()
        .map(|d| p.clone().eval(d))
        .collect();
    let mt = MerkleTree::new(ev.clone());
    let ch = Channel::new();
    (t, g, points, h_gen, h, domain, p, ev, mt, ch)
}


pub fn part2() -> (Polynomial, Vec<FieldElement>, MerkleTree, Channel, Vec<FieldElement>) {
    let (_t, _g, points, _h_gen, _h, domain, p, _ev, _mt, ch) = part1();
    let numer0 = p.clone() - Polynomial::new(&[FieldElement::one()]);
    let denom0 = Polynomial::gen_linear_term(FieldElement::one());
    let (q0, _r0) = numer0.qdiv(denom0);
    let numer1 = p.clone() - Polynomial::new(&[FieldElement::new(2338775057)]);
    let denom1 = Polynomial::gen_linear_term(points[1022]);
    let (q1, _r1) = numer1.qdiv(denom1);
    let inner_poly0 = Polynomial::new(&[FieldElement::zero(), points[2]]);
    let final0 = p.clone().compose(inner_poly0);

    let inner_poly1 = Polynomial::new(&[FieldElement::zero(), points[1]]);
    let composition = p.clone().compose(inner_poly1);
    let final1 = composition.clone() * composition;
    let final2 = p.clone() * p;
    let numer2 = final0 - final1 - final2;
    let mut coef = vec![FieldElement::one()];
    for _ in 0..1023 {
        coef.push(FieldElement::zero());
    }
    coef.push(FieldElement(FieldElement::k_modulus() - 1));
    let numerator_of_denom2 = Polynomial::new(&coef);
    let factor0 = Polynomial::gen_linear_term(points[1021]);
    let factor1 = Polynomial::gen_linear_term(points[1022]);
    let factor2 = Polynomial::gen_linear_term(points[1023]);
    let denom_of_denom2 = factor0 * factor1 * factor2;
    let (denom2, _r_denom2) = numerator_of_denom2.qdiv(denom_of_denom2);
    let (q2, _r2) = numer2.qdiv(denom2);
    let mut ch_mut = ch.clone();
    let cp0 = q0 * ch_mut.receive_random_field_element();
    let cp1 = q1 * ch_mut.receive_random_field_element();
    let cp2 = q2 * ch_mut.receive_random_field_element();

    let cp = cp0 + cp1 + cp2;
    let cp_ev: Vec<FieldElement> = domain.clone().into_iter().map(|d| cp(d)).collect();
    let cp_mt = MerkleTree::new(cp_ev.clone());
    ch_mut.send(cp_mt.root());

    (cp, cp_ev, cp_mt, ch_mut, domain)  
}
