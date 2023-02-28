/// A Channel instance can be used by a prover or a verifier to preserve the semantics of an
/// interactive proof system, while under the hood it is in fact non-interactive, and uses Sha256
/// to generate randomness when this is required.
/// It allows writing string-form data to it, and reading either random integers of random
/// FieldElements from it.
pub struct Channel; // {
