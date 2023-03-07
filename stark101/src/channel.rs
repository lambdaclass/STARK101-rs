use crate::field::FieldElement;
use sha256;

/// A Channel instance can be used by a prover or a verifier to preserve the semantics of an
/// interactive proof system, while under the hood it is in fact non-interactive, and uses Sha256
/// to generate randomness when this is required.
/// It allows writing string-form data to it, and reading either random integers of random
/// FieldElements from it.
pub struct Channel {
    proof: Vec<String>,
    state: String,
}

impl Channel {
    pub fn new() -> Channel {
        Channel {
            state: "0".to_string(),
            proof: vec![],
        }
    }

    pub fn send(&mut self, s: String) {
        let current_state = self.state.clone();
        self.state = sha256::digest(current_state + &s);
        self.proof.push(format!("{s}"));
    }

    /// Emulates a random field element sent by the verifier.
    pub fn receive_random_field_element(&mut self) -> FieldElement {
        let num = self.receive_random_int(0, FieldElement::k_modulus() - 1, false);
        self.proof.push(format!("{num}"));
        FieldElement::new(num)
    }

    /// Emulates a random integer sent by the verifier in the range [min, max]
    /// (including min and max).
    pub fn receive_random_int(&mut self, min: usize, max: usize, show_in_proof: bool) -> usize {
        // Note that when the range is close to 2^256 this does not emit a uniform distribution,
        // even if sha256 is uniformly distributed.
        // It is, however, close enough for this tutorial's purposes.
        dbg!(&self.state);
        let num = min + (usize::from_str_radix(&self.state, 16).unwrap() % (max - min + 1));
        let current_state = self.state.clone();
        self.state = sha256::digest(current_state);
        if show_in_proof {
            self.proof.push(format!("{num}"));
        }
        num
    }
}
