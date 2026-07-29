use rand::{Rng as _, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

/// Stable, deterministic per-game RNG.
///
/// ChaCha8 is fast enough for a 108-tile shuffle and, unlike a thread-local
/// generator, keeps replay output independent of Rayon scheduling.
#[derive(Clone, Debug)]
pub(crate) struct Rng(ChaCha8Rng);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(ChaCha8Rng::seed_from_u64(seed))
    }

    #[inline]
    pub(crate) fn bounded(&mut self, upper: u32) -> u32 {
        self.0.random_range(0..upper)
    }

    #[inline]
    pub(crate) fn unit_f64(&mut self) -> f64 {
        self.0.random()
    }

    pub(crate) fn shuffle(&mut self, values: &mut [u8]) {
        values.shuffle(&mut self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn shuffle_is_deterministic_and_preserves_the_deck() {
        let original: Vec<u8> = (0..108).collect();
        let mut first = original.clone();
        let mut second = original.clone();
        Rng::new(7).shuffle(&mut first);
        Rng::new(7).shuffle(&mut second);

        assert_eq!(first, second);
        assert_ne!(first, original);
        first.sort_unstable();
        assert_eq!(first, original);
    }

    #[test]
    fn bounded_values_stay_in_range() {
        let mut rng = Rng::new(42);
        for upper in [1, 2, 3, 27, 108, u32::MAX] {
            for _ in 0..1_000 {
                assert!(rng.bounded(upper) < upper);
            }
        }
    }
}
