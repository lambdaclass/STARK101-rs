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
    let mt = MerkleTree;
    let ch = Channel::new();
    (t, g, points, h_gen, h, domain, p, ev, mt, ch)
}
