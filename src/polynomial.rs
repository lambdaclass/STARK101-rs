use crate::field::FieldElement;
use itertools::{EitherOrBoth, Itertools, enumerate};

pub fn x() -> Polynomial {
    Polynomial::X()
}

/// Represents a polynomial over FieldElement.
#[derive(Debug, PartialEq, Clone)]
pub struct Polynomial(Vec<FieldElement>);

impl Polynomial {
    /// Creates a new Polynomial with the given coefficients.
    /// Internally storing the coefficients in self.poly, least-significant (i.e. free term)
    /// first, so 9 - 3x^2 + 19x^5 is represented internally by the vector [9, 0, -3, 0, 0, 19].
    pub fn new(coefficients: &[FieldElement]) -> Self {
        Polynomial(coefficients.into())
    }

    /// Returns the polynomial x.
    pub fn X() -> Self {
        Polynomial(vec![FieldElement::zero(), FieldElement::one()])
    }

    /// Constructs the monomial coefficient * x^degree.
    pub fn monomial(degree: usize, coefficient: FieldElement) -> Self {
        let mut coefficients = [FieldElement::zero()].repeat(degree);
        coefficients.push(coefficient);
        Polynomial::new(&coefficients)
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
    pub fn degree(&self) -> usize {
        Self::trim_trailing_zeros(&self.0).len() - 1
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
                EitherOrBoth::Both(e1, e2) => operation(*e1, *e2),
                EitherOrBoth::Left(e) => operation(*e, fill_value),
                EitherOrBoth::Right(e) => operation(*e, fill_value),
            })
            .collect()
    }

    /// Returns the coefficient of x^n
    pub fn get_nth_degree_coefficient(&self, n: usize) -> FieldElement {
        if n > self.degree() {
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
    pub fn eval(&self, point: FieldElement) -> Self {
        todo!()
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

    pub fn calculate_lagrange_polynomials(x_values: &[FieldElement]) -> Vec<Self> {
        todo!()
    }

    pub fn interpolate_poly_lagrange(
        y_values: &[FieldElement],
        lagrange_polynomials: Vec<Self>,
    ) -> Self {
        todo!()
    }

    /// Returns q, r the quotient and remainder polynomials respectively, such that
    /// f = q * g + r, where deg(r) < deg(g).
    /// * Assert that g is not the zero polynomial.
    fn qdiv(&self, other: impl Into<Polynomial>) -> (Polynomial, Polynomial) {
        let other_poly: Polynomial = other.into();
        let other_elems = Polynomial::trim_trailing_zeros(&other_poly.0);
        assert!(!other_elems.is_empty(), "Dividing by zero polynomial.");
        let self_elems = Polynomial::trim_trailing_zeros(&self.0);
        if self_elems.is_empty() {
            return (Polynomial(vec![]), Polynomial(vec![]))
        }

        let mut rem = self_elems.clone();
        let mut degree_difference = rem.len() - other_elems.len();
        let mut quotient = vec![FieldElement::zero()].repeat(degree_difference + 1).to_vec();
        while degree_difference >= 0 {

            let mut tmp = rem.last().unwrap().to_owned() * other_elems.last().unwrap().inverse();
            quotient[degree_difference] = quotient[degree_difference] + tmp;
            let mut last_non_zero = degree_difference - 1;
            for (i, coef) in enumerate(other_elems.clone()) {
                let i = i + degree_difference;
                rem[i] = rem[i] - (tmp * coef);
                if rem[i] != FieldElement::zero() {
                    last_non_zero = i
                }
            }
            // Eliminate trailing zeroes (i.e. make r end with its last non-zero coefficient).
            rem = rem.into_iter().take(last_non_zero + 1).collect();
            degree_difference = rem.len() - other_elems.len();
        }

        (Polynomial(Self::trim_trailing_zeros(&quotient)), Polynomial(rem))
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
        let other_poly: Polynomial = other.into();
        self + other_poly
    }
}

impl std::ops::Sub<FieldElement> for Polynomial {
    type Output = Polynomial;

    fn sub(self, other: FieldElement) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self + other_poly
    }
}

impl std::ops::Neg for Polynomial {
    type Output = Polynomial;

    fn neg(self) -> Self::Output {
        Polynomial(vec![]) - self
    }
}

impl std::ops::Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: Self) -> Self::Output {
        let mut res = [FieldElement::zero()].repeat(self.degree() + other.degree() + 1);
        for (i, c1) in self.0.into_iter().enumerate() {
            for (j, c2) in other.clone().0.into_iter().enumerate() {
                res[i + j] += c1 * c2;
            }
        }
        Polynomial(res)
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
        assert!(rem == 0, "Polynomials are not divisible."); 
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

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use crate::field::FieldElement;
    use super::{Polynomial, x};

    /// Returns a random polynomial of a prescribed degree which is not the zero polynomial.
    fn generate_random_polynomail(degree: usize) -> Polynomial {
        let leading = FieldElement::random_element(&[FieldElement::zero()]);
        let mut elems = (1..degree).into_iter().map(|_| FieldElement::random_element(&[])).collect_vec();
        elems.push(leading);
        Polynomial(elems)
    }
    
    #[test]
    fn test_poly_mul() {
        let result = (x() + 1) * (x() + 1);
        let expected = x().pow(2) + x()*2usize + 1;
        assert_eq!(result, expected)
    }

    #[test]
    fn test_div() {
        let p = x().pow(2) - 1;
        assert_eq!(p / (x() - 1), x() + 1)
    }

    #[test]
    fn test_modulo() {
        let p: Polynomial = x().pow(9) - x() * 5usize + 4;
        assert_eq!(p.modulo(x().pow(2) + 1), x() * (-4i128) + 4)
    }
}
