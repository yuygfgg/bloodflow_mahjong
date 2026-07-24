use std::hint::black_box;
use std::time::{Duration, Instant};

use bloodflow_mahjong::{Meld, Suit, Tile, evaluate_max_wait};

fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank - 1).unwrap()
}

fn add(counts: &mut [u8; 27], suit: Suit, rank: u8, amount: u8) {
    counts[tile(suit, rank).index()] += amount;
}

fn measure(name: &str, counts: &[u8; 27], melds: &[Meld], iterations: u32) {
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(evaluate_max_wait(black_box(counts), black_box(melds), None));
    }
    let elapsed = start.elapsed();
    let each = elapsed / iterations;
    println!("{name}: {each:?}/call ({iterations} iterations, {elapsed:?})");
}

fn main() {
    let mut normal = [0; 27];
    for rank in 1..=4 {
        add(&mut normal, Suit::Characters, rank, 3);
    }
    add(&mut normal, Suit::Characters, 5, 1);

    let mut expanded = [0; 27];
    add(&mut expanded, Suit::Characters, 2, 4);
    add(&mut expanded, Suit::Bamboo, 5, 4);
    add(&mut expanded, Suit::Dots, 8, 4);
    add(&mut expanded, Suit::Bamboo, 2, 1);
    add(&mut expanded, Suit::Characters, 1, 1);

    let dense = [4; 27];
    let dense_triplets = [3; 27];
    black_box(evaluate_max_wait(&normal, &[], None));
    std::thread::sleep(Duration::from_millis(10));
    measure("normal", &normal, &[], 1_000);
    measure("expanded", &expanded, &[], 100);
    measure("dense", &dense, &[], 3);
    measure("dense-triplets", &dense_triplets, &[], 1);
}
