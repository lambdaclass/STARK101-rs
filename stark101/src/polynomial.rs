use crate::field::FieldElement;
use itertools::{enumerate, EitherOrBoth, Itertools};

pub fn x() -> Polynomial {
    Polynomial::x()
}

/// Represents a polynomial over FieldElement.
#[derive(Debug, PartialEq, Clone)]
pub struct Polynomial(pub Vec<FieldElement>);

impl Polynomial {
    /// Creates a new Polynomial with the given coefficients.
    /// Internally storing the coefficients in self.poly, least-significant (i.e. free term)
    /// first, so 9 - 3x^2 + 19x^5 is represented internally by the vector [9, 0, -3, 0, 0, 19].
    pub fn new(coefficients: &[FieldElement]) -> Self {
        Polynomial(coefficients.into())
    }

    /// Returns the polynomial x.
    pub fn x() -> Self {
        Polynomial(vec![FieldElement::zero(), FieldElement::one()])
    }

    /// Constructs the monomial coefficient * x^degree.
    pub fn monomial(degree: usize, coefficient: FieldElement) -> Self {
        let mut coefficients = [FieldElement::zero()].repeat(degree);
        coefficients.push(coefficient);
        Polynomial::new(&coefficients)
    }

    /// Computes the product of the given polynomials.
    pub fn prod(values: &[Polynomial]) -> Polynomial {
        let len_values = values.len();
        if len_values == 0 {
            return Polynomial(vec![FieldElement::one()]);
        };
        if len_values == 1 {
            return values.first().unwrap().to_owned().into();
        };
        let prod_left = values
            .into_iter()
            .take(len_values / 2)
            .map(ToOwned::to_owned)
            .collect_vec();
        let prod_right = values
            .into_iter()
            .skip(len_values / 2)
            .map(ToOwned::to_owned)
            .collect_vec();
        Self::prod(&prod_left) * Self::prod(&prod_right)
    }

    /// Generates the polynomial (x-p) for a given point p.
    pub fn gen_linear_term(point: FieldElement) -> Self {
        Polynomial::new(&[FieldElement::zero() - point, FieldElement::one()])
    }

    pub fn modulo(&self, other: Polynomial) -> Polynomial {
        self.qdiv(other).1
    }

    /// The polynomials are represented by a list so the degree is the length of the list minus the
    /// number of trailing zeros (if they exist) minus 1.
    /// This implies that the degree of the zero polynomial will be -1.
    pub fn degree(&self) -> isize {
        Self::trim_trailing_zeros(&self.0).len() as isize - 1
    }

    fn remove_trailing_elements(
        elements: &[FieldElement],
        element_to_remove: &FieldElement,
    ) -> Vec<FieldElement> {
        let it = elements
            .into_iter()
            .rev()
            .skip_while(|x| *x == element_to_remove)
            .map(Clone::clone);
        let mut v = it.collect::<Vec<FieldElement>>();
        v.reverse();
        v
    }

    fn scalar_operation<F>(
        elements: &[FieldElement],
        operation: F,
        scalar: impl Into<FieldElement>,
    ) -> Vec<FieldElement>
    where
        F: Fn(FieldElement, FieldElement) -> FieldElement,
    {
        let value: FieldElement = scalar.into();
        elements.into_iter().map(|e| operation(*e, value)).collect()
    }

    /// Removes zeros from the end of a list.
    fn trim_trailing_zeros(p: &[FieldElement]) -> Vec<FieldElement> {
        Self::remove_trailing_elements(p, &FieldElement::zero())
    }

    fn two_list_tuple_operation<F>(
        l1: &[FieldElement],
        l2: &[FieldElement],
        operation: F,
        fill_value: FieldElement,
    ) -> Vec<FieldElement>
    where
        F: Fn(FieldElement, FieldElement) -> FieldElement,
    {
        l1.into_iter()
            .zip_longest(l2)
            .map(|x| match x {
                EitherOrBoth::Both(e1, e2) => operation(e1.to_owned(), e2.to_owned()),
                EitherOrBoth::Left(e) => operation(e.to_owned(), fill_value),
                EitherOrBoth::Right(e) => operation(e.to_owned(), fill_value),
            })
            .collect()
    }

    /// Returns the coefficient of x^n
    pub fn get_nth_degree_coefficient(&self, n: usize) -> FieldElement {
        if n > self.degree() as _ {
            FieldElement::zero()
        } else {
            self.0[n]
        }
    }

    /// Multiplies polynomial by a scalar.
    pub fn scalar_mul(&self, scalar: usize) -> Self {
        Polynomial(Self::scalar_operation(&self.0, |x, y| x * y, scalar))
    }

    /// Evaluates the polynomial at the given point using Horner evaluation.
    pub fn eval(&self, point: impl Into<FieldElement>) -> FieldElement {
        let point: FieldElement = point.into();
        let mut val = FieldElement::zero();
        for coef in self.0.clone().into_iter().rev() {
            val = val * point + coef;
        }
        val
    }

    /// Calculates self^other using repeated squaring.
    pub fn pow(&self, other: usize) -> Self {
        let mut other = other;
        let mut res = Polynomial(vec![FieldElement::one()]);
        let mut current = self.to_owned();
        loop {
            if other % 2 != 0 {
                res = res * current.to_owned();
            }
            other >>= 1;
            if other == 0 {
                break;
            }
            current = current.to_owned() * current;
        }
        res
    }

    /// Given the x_values for evaluating some polynomials,
    /// it computes part of the lagrange polynomials required to interpolate a polynomial over this domain.
    pub fn calculate_lagrange_polynomials(x_values: &[FieldElement]) -> Vec<Self> {
        let mut lagrange_polynomials = vec![];
        let monomials = x_values
            .into_iter()
            .map(|x| Self::monomial(1, FieldElement::one()) - Self::monomial(0, *x))
            .collect_vec();
        let numerator = Self::prod(&monomials);
        for j in 0..(x_values.len()) {
            // In the denominator, we have:
            // (x_j-x_0)(x_j-x_1)...(x_j-x_{j-1})(x_j-x_{j+1})...(x_j-x_{len(X)-1})
            let denominator_values = enumerate(x_values)
                .filter(|(i, _)| *i != j)
                .map(|(_, x)| {
                    let poly: Polynomial = x_values[j].into();
                    let x_poly: Polynomial = (*x).into();
                    poly - x_poly
                })
                .collect_vec();
            let denominator = Polynomial::prod(&denominator_values);
            // Numerator is a bit more complicated, since we need to compute a poly multiplication here.
            //  Similarly to the denominator, we have:
            // (x-x_0)(x-x_1)...(x-x_{j-1})(x-x_{j+1})...(x-x_{len(X)-1})
            let (cur_poly, _) = numerator.qdiv(monomials[j].clone() * denominator);
            lagrange_polynomials.push(cur_poly);
        }

        lagrange_polynomials
    }

    /// Interpolate the polynomials given a set of y_values.
    /// - y_values: y coordinates of the points.
    /// - lagrange_polynomials: the polynomials obtained from calculate_lagrange_polynomials.
    ///
    /// Returns the interpolated polynomial.
    pub fn interpolate_poly_lagrange(
        y_values: &[FieldElement],
        lagrange_polynomials: Vec<Self>,
    ) -> Self {
        let mut poly = Polynomial(vec![]);
        for (j, y_value) in enumerate(y_values) {
            poly = poly + lagrange_polynomials[j].scalar_mul((*y_value).into());
        }
        poly
    }

    /// Returns q, r the quotient and remainder polynomials respectively, such that
    /// f = q * g + r, where deg(r) < deg(g).
    /// * Assert that g is not the zero polynomial.
    pub fn qdiv(&self, other: impl Into<Polynomial>) -> (Polynomial, Polynomial) {
        let other_poly: Polynomial = other.into();
        let other_elems = Polynomial::trim_trailing_zeros(&other_poly.0);
        assert!(!other_elems.is_empty(), "Dividing by zero polynomial.");
        let self_elems = Polynomial::trim_trailing_zeros(&self.0);
        if self_elems.is_empty() {
            return (Polynomial(vec![]), Polynomial(vec![]));
        }

        let mut rem = self_elems.clone();
        let mut degree_difference = rem.len() as isize - other_elems.len() as isize;
        let mut quotient = if degree_difference > 0 {
            vec![FieldElement::zero()]
                .repeat((degree_difference + 1) as usize)
                .to_vec()
        } else {
            vec![FieldElement::zero()]
        };
        while degree_difference >= 0 {
            let tmp = rem.last().unwrap().to_owned() * other_elems.last().unwrap().inverse();
            quotient[degree_difference as usize] = quotient[degree_difference as usize] + tmp;
            let mut last_non_zero = degree_difference as isize - 1;
            for (i, coef) in enumerate(other_elems.clone()) {
                let k = i + degree_difference as usize;
                rem[k] = rem[k] - (tmp * coef);
                if rem[k] != FieldElement::zero() {
                    last_non_zero = k as isize
                }
            }
            // Eliminate trailing zeroes (i.e. make r end with its last non-zero coefficient).
            rem = rem.into_iter().take((last_non_zero + 1) as usize).collect();
            degree_difference = rem.len() as isize - other_elems.len() as isize;
        }

        (
            Polynomial(Self::trim_trailing_zeros(&quotient)),
            Polynomial(rem),
        )
    }

    /// Returns a polynomial of degree < len(x_values) that evaluates to y_values[i] on x_values[i] for all i.
    pub fn interpolate(x_values: &[FieldElement], y_values: &[FieldElement]) -> Polynomial {
        assert!(x_values.len() == y_values.len());
        let lp = Self::calculate_lagrange_polynomials(x_values);
        Self::interpolate_poly_lagrange(y_values, lp)
    }

    // Composes this polynomial with `other`.
    // Example:
    // f = x().pow(2) + x()
    // g = x() + 1
    // f.compose(g) == (2 + x()*3 + x().pow(2))
    pub fn compose(&self, other: impl Into<Polynomial>) -> Polynomial {
        let other_poly: Polynomial = other.into();
        let mut res = Polynomial(vec![]);
        for coef in self.0.clone().into_iter().rev() {
            res = (res * other_poly.clone()) + Polynomial(vec![coef]);
        }
        res
    }
}

impl FnOnce<(Polynomial,)> for Polynomial {
    type Output = Polynomial;

    extern "rust-call" fn call_once(self, args: (Polynomial,)) -> Self::Output {
        self.compose(args.0)
    }
}

impl FnMut<(Polynomial,)> for Polynomial {
    extern "rust-call" fn call_mut(&mut self, args: (Polynomial,)) -> Self::Output {
        self.compose(args.0)
    }
}

impl Fn<(Polynomial,)> for Polynomial {
    extern "rust-call" fn call(&self, args: (Polynomial,)) -> Self::Output {
        self.compose(args.0)
    }
}

impl FnOnce<(FieldElement,)> for Polynomial {
    type Output = FieldElement;

    extern "rust-call" fn call_once(self, args: (FieldElement,)) -> Self::Output {
        self.eval(args.0)
    }
}

impl FnMut<(FieldElement,)> for Polynomial {
    extern "rust-call" fn call_mut(&mut self, args: (FieldElement,)) -> Self::Output {
        self.eval(args.0)
    }
}

impl Fn<(FieldElement,)> for Polynomial {
    extern "rust-call" fn call(&self, args: (FieldElement,)) -> Self::Output {
        self.eval(args.0)
    }
}

impl FnOnce<(i128,)> for Polynomial {
    type Output = FieldElement;

    extern "rust-call" fn call_once(self, args: (i128,)) -> Self::Output {
        let fe: FieldElement = args.0.into();
        self.eval(fe)
    }
}

impl FnMut<(i128,)> for Polynomial {
    extern "rust-call" fn call_mut(&mut self, args: (i128,)) -> Self::Output {
        let fe: FieldElement = args.0.into();
        self.eval(fe)
    }
}

impl Fn<(i128,)> for Polynomial {
    extern "rust-call" fn call(&self, args: (i128,)) -> Self::Output {
        let fe: FieldElement = args.0.into();
        self.eval(fe)
    }
}

impl PartialEq<usize> for Polynomial {
    fn eq(&self, other: &usize) -> bool {
        let fe: FieldElement = (*other).into();
        let poly: Polynomial = fe.into();
        self == &poly
    }
}

impl PartialEq<FieldElement> for Polynomial {
    fn eq(&self, other: &FieldElement) -> bool {
        let other_poly: Polynomial = (*other).into();
        self == &other_poly
    }
}

impl From<usize> for Polynomial {
    fn from(value: usize) -> Self {
        let fe: FieldElement = value.into();
        fe.into()
    }
}

impl From<FieldElement> for Polynomial {
    fn from(value: FieldElement) -> Self {
        Polynomial::new(&[value])
    }
}

impl std::ops::Add for Polynomial {
    type Output = Polynomial;

    fn add(self, other: Self) -> Self::Output {
        Polynomial(Self::two_list_tuple_operation(
            &self.0,
            &other.0,
            |x, y| x + y,
            FieldElement::zero(),
        ))
    }
}

impl std::ops::Add<usize> for Polynomial {
    type Output = Polynomial;

    fn add(self, other: usize) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self + other_poly
    }
}

impl std::ops::Add<FieldElement> for Polynomial {
    type Output = Polynomial;

    fn add(self, other: FieldElement) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self + other_poly
    }
}

impl std::ops::Sub for Polynomial {
    type Output = Polynomial;

    fn sub(self, other: Self) -> Self::Output {
        Polynomial(Self::two_list_tuple_operation(
            &self.0,
            &other.0,
            |x, y| x - y,
            FieldElement::zero(),
        ))
    }
}

impl std::ops::Sub<usize> for Polynomial {
    type Output = Polynomial;

    fn sub(self, other: usize) -> Self::Output {
        let other_fe: FieldElement = (-(other as i128)).into();
        let other_poly: Polynomial = other_fe.into();
        self + other_poly
    }
}

impl std::ops::Sub<FieldElement> for Polynomial {
    type Output = Polynomial;

    fn sub(self, other: FieldElement) -> Self::Output {
        let other_poly: Polynomial = other.into();

        self - other_poly
    }
}

impl std::ops::Neg for Polynomial {
    type Output = Polynomial;

    fn neg(self) -> Self::Output {
        Polynomial(vec![]) - self
    }
}

impl std::ops::Mul<Polynomial> for usize {
    type Output = Polynomial;

    fn mul(self, rhs: Polynomial) -> Self::Output {
        rhs * self
    }
}

impl std::ops::Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: Self) -> Self::Output {
        let mut res = [FieldElement::zero()].repeat((self.degree() + other.degree() + 1) as usize);
        for (i, c1) in self.0.clone().into_iter().enumerate() {
            if other.degree() > -1 {
                for (j, c2) in other.clone().0.into_iter().enumerate() {
                    if let Some(value) = res.get_mut(i + j) {
                        *value += c1 * c2;
                    }
                }
            }
        }
        Polynomial(Self::trim_trailing_zeros(&res))
    }
}

impl std::ops::Mul<usize> for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: usize) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self * other_poly
    }
}

impl std::ops::Mul<i128> for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: i128) -> Self::Output {
        let other_fe: FieldElement = other.into();
        let other_poly: Polynomial = other_fe.into();
        self * other_poly
    }
}

impl std::ops::Mul<FieldElement> for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: FieldElement) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self * other_poly
    }
}

impl std::ops::Div for Polynomial {
    type Output = Polynomial;

    fn div(self, other: Self) -> Self::Output {
        let (div, rem) = self.qdiv(other);

        assert!(rem.0.is_empty(), "Polynomials are not divisible.");
        div
    }
}

impl std::ops::Div<usize> for Polynomial {
    type Output = Polynomial;

    fn div(self, other: usize) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self / other_poly
    }
}

impl std::ops::Div<FieldElement> for Polynomial {
    type Output = Polynomial;

    fn div(self, other: FieldElement) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self / other_poly
    }
}

impl std::ops::Rem for Polynomial {
    type Output = Polynomial;

    fn rem(self, other: Self) -> Self::Output {
        let (_, remainder) = self.qdiv(other);
        remainder
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{x, Polynomial};
    use crate::{field::FieldElement, parts};
    use itertools::Itertools;
    use rand::Rng;

    /// Returns a random polynomial of a prescribed degree which is not the zero polynomial.
    fn generate_random_polynomial(degree: usize) -> Polynomial {
        let leading = FieldElement::random_element(&[FieldElement::zero()]);
        let mut elems = (0..degree)
            .into_iter()
            .map(|_| FieldElement::random_element(&[]))
            .collect_vec();
        elems.push(leading);
        Polynomial(elems)
    }

    #[test]
    fn test_generate_polynomial() {
        let generated_poly = generate_random_polynomial(5);
        assert_eq!(5, generated_poly.degree())
    }

    #[test]
    fn test_poly_mul() {
        let result = (x() + 1) * (x() + 1);
        let expected = x().pow(2) + x() * 2usize + 1;
        assert_eq!(result, expected)
    }

    #[test]
    fn test_poly_mul_empty() {
        let empty_poly = Polynomial::new(&[]);
        let result = empty_poly.clone() * FieldElement::new(3) * (x() + 1)
            + empty_poly.clone() * Polynomial::prod(&[1.into(), 2.into(), 100.into()]);
        let expected = empty_poly;
        assert_eq!(result, expected)
    }

    #[test]
    fn test_div() {
        let p = x().pow(2) - FieldElement::one();
        assert_eq!(p / (x() - FieldElement::one()), x() + FieldElement::one())
    }

    #[test]
    fn test_modulo() {
        let p: Polynomial = x().pow(9) - (x() * 5usize) + 4;
        assert_eq!(p.modulo(x().pow(2) + 1), x() * (-4i128) + 4)
    }

    #[test]
    fn test_prod() {
        let g = FieldElement::generator().pow((FieldElement::k_modulus() - 1) / 1024);
        let polys = (0..1024).into_iter().map(|i| x() - g.pow(i)).collect_vec();
        assert_eq!(
            x().pow(1024) - FieldElement::one(),
            Polynomial::prod(&polys)
        )
    }

    #[test]
    fn test_call_compose() {
        let p = x().pow(2) + x();
        assert_eq!(p(x() + 1), x().pow(2) + x() * 3usize + 2)
    }

    #[test]
    fn test_call_eval() {
        let p = x().pow(2) + x();
        assert_eq!(p(5), 30)
    }

    #[test]
    fn test_div_rand_poly() {
        let iterations = 20;
        let mut rng = rand::thread_rng();
        for _ in 0..iterations {
            let degree_a = rng.gen_range(0..50);
            let degree_b = rng.gen_range(0..50);
            let poly_a = generate_random_polynomial(degree_a);
            let poly_b = generate_random_polynomial(degree_b);
            let (q, r) = poly_a.qdiv(poly_b.clone());
            let d = r.clone() + q * poly_b.clone();
            assert!(r.degree() < poly_b.degree());
            assert_eq!(d, poly_a);
        }
    }

    #[test]
    fn test_poly_interpolation() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let degree = rng.gen_range(0..100);
            let p = generate_random_polynomial(degree);
            let mut x_values_set: HashSet<_> = HashSet::new();
            // Evaluate it on a number of points that is at least its degree.
            while x_values_set.len() < degree + 1 {
                x_values_set.insert(FieldElement::random_element(&[]));
            }
            let x_values: Vec<FieldElement> = x_values_set.into_iter().collect_vec();
            let y_values = x_values
                .clone()
                .into_iter()
                .map(|x| p.eval(x))
                .collect_vec();
            // Obtain a polynomial from the evaluation.
            let interpolated_p = Polynomial::interpolate(&x_values, &y_values);
            assert_eq!(p, interpolated_p)
        }
    }

    #[test]
    fn test_compose() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let outer_poly = generate_random_polynomial(rng.gen_range(0..1024));
            let inner_poly = generate_random_polynomial(rng.gen_range(0..16));
            // Validate th evaluation of the composition poly outer_poly(inner_poly) on a random point.
            let point = FieldElement::random_element(&[]);
            let result = outer_poly.clone()(inner_poly.clone()).eval(point);
            let expected = outer_poly.eval(inner_poly.eval(point));
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_polynomial_pow() {
        let (_, g, _, _, _, _, f, _, _, _) = parts::part1();
        let numer_1 = f(x() * g.pow(2));
        let numer_2 = f(x() * g).pow(2)
            * FieldElement::new((-1 + FieldElement::k_modulus() as i128) as usize);
        let numer_3 =
            f.pow(2) * FieldElement::new((-1 + FieldElement::k_modulus() as i128) as usize);
        let numer = numer_1 + numer_2 + numer_3;
        assert_eq!(FieldElement::new(0), numer(g.pow(1020)));
        assert_eq!(FieldElement::new(230576507), numer(g.pow(1021)));
    }

    #[test]
    fn test_compose_session2() {
        let q = 2 * x().pow(2) + 1;
        let r = x() - 3;
        let expected = (2 * x().pow(2)) - 12 * x() + 19;

        assert_eq!(expected, q(r));
    }
}
