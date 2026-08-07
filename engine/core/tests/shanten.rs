use bloodflow_mahjong::{
    Batch, GameError, Meld, MeldKind, SHANTEN_COMPLETE, SHANTEN_TERMINAL, Seat, Suit, Tile,
    analyze_shanten, evaluate_shanten, is_winning,
};

type Counts = [u8; 27];

fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank - 1).expect("test ranks are in 1..=9")
}

fn hand(tiles: &[(Suit, u8, u8)]) -> Counts {
    let mut counts = [0; 27];
    for &(suit, rank, count) in tiles {
        counts[tile(suit, rank).index()] += count;
    }
    counts
}

fn meld(suit: Suit, rank: u8) -> Meld {
    Meld {
        tile: tile(suit, rank),
        kind: MeldKind::Pong,
        source: Seat::EAST,
    }
}

#[test]
fn standard_ready_hand_reports_its_winning_tile() {
    let counts = hand(&[
        (Suit::Characters, 1, 1),
        (Suit::Characters, 2, 1),
        (Suit::Characters, 3, 1),
        (Suit::Bamboo, 1, 1),
        (Suit::Bamboo, 2, 1),
        (Suit::Bamboo, 3, 1),
        (Suit::Dots, 1, 1),
        (Suit::Dots, 2, 1),
        (Suit::Dots, 3, 1),
        (Suit::Dots, 7, 3),
        (Suit::Characters, 5, 1),
    ]);
    let analysis = analyze_shanten(&counts, &[], None);
    assert_eq!(analysis.shanten, 0);
    assert_eq!(
        analysis.improving_tiles,
        1 << tile(Suit::Characters, 5).index()
    );

    let mut complete = counts;
    complete[tile(Suit::Characters, 5).index()] += 1;
    assert_eq!(evaluate_shanten(&complete, &[], None), SHANTEN_COMPLETE);
}

#[test]
fn exposed_melds_reduce_the_number_of_concealed_groups_needed() {
    let melds = [meld(Suit::Characters, 1)];
    let counts = hand(&[
        (Suit::Bamboo, 1, 1),
        (Suit::Bamboo, 2, 1),
        (Suit::Bamboo, 3, 1),
        (Suit::Dots, 1, 1),
        (Suit::Dots, 2, 1),
        (Suit::Dots, 3, 1),
        (Suit::Dots, 7, 3),
        (Suit::Characters, 5, 1),
    ]);
    let analysis = analyze_shanten(&counts, &melds, None);
    assert_eq!(analysis.shanten, 0);
    assert!(analysis.improving_tiles & (1 << tile(Suit::Characters, 5).index()) != 0);
}

#[test]
fn exposed_melds_share_the_four_copy_limit_with_concealed_tiles() {
    let blocked_tile = tile(Suit::Characters, 5);
    let melds = [
        meld(Suit::Characters, 5),
        meld(Suit::Bamboo, 1),
        meld(Suit::Dots, 3),
        meld(Suit::Dots, 7),
    ];
    let counts = hand(&[(Suit::Characters, 5, 2)]);

    assert_ne!(
        evaluate_shanten(&counts, &melds, None),
        SHANTEN_COMPLETE,
        "a pong and a concealed pair cannot represent five physical copies"
    );
    assert!(!is_winning(&counts, &melds, Some(blocked_tile)));

    let ready_counts = hand(&[(Suit::Characters, 5, 1)]);
    let analysis = analyze_shanten(&ready_counts, &melds, None);
    assert_eq!(analysis.shanten, 0);
    assert_eq!(analysis.improving_tiles & (1 << blocked_tile.index()), 0);
}

#[test]
fn seven_pairs_counts_a_quad_as_two_pairs() {
    let counts = hand(&[
        (Suit::Characters, 1, 4),
        (Suit::Characters, 2, 4),
        (Suit::Characters, 3, 4),
        (Suit::Characters, 4, 1),
    ]);
    let analysis = analyze_shanten(&counts, &[], None);
    assert_eq!(analysis.shanten, 0);
    assert!(analysis.improving_tiles & (1 << tile(Suit::Characters, 4).index()) != 0);

    let mut complete = counts;
    complete[tile(Suit::Characters, 4).index()] += 1;
    assert_eq!(evaluate_shanten(&complete, &[], None), SHANTEN_COMPLETE);
}

#[test]
fn missing_suit_tiles_cannot_form_shapes_or_improvements() {
    let counts = hand(&[
        (Suit::Characters, 1, 1),
        (Suit::Characters, 2, 1),
        (Suit::Characters, 3, 1),
        (Suit::Bamboo, 1, 1),
        (Suit::Bamboo, 2, 1),
        (Suit::Bamboo, 3, 1),
        (Suit::Dots, 1, 1),
        (Suit::Dots, 2, 1),
        (Suit::Dots, 3, 1),
        (Suit::Dots, 7, 3),
        (Suit::Characters, 5, 1),
    ]);
    let unrestricted = analyze_shanten(&counts, &[], None);
    let restricted = analyze_shanten(&counts, &[], Some(Suit::Characters));
    assert_eq!(unrestricted.shanten, 0);
    assert!(restricted.shanten > unrestricted.shanten);
    assert_eq!(restricted.improving_tiles & Suit::Characters.mask(), 0);
}

#[test]
fn complete_shanten_matches_the_win_evaluator_for_exact_hands() {
    let mut random = 0x68b1_2f43_95ce_d70a_u64;
    for _ in 0..2_000 {
        let mut counts = [0_u8; 27];
        for _ in 0..14 {
            loop {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let index = random as usize % counts.len();
                if counts[index] < 4 {
                    counts[index] += 1;
                    break;
                }
            }
        }
        assert_eq!(
            evaluate_shanten(&counts, &[], None) == SHANTEN_COMPLETE,
            is_winning(&counts, &[], None),
            "counts={counts:?}"
        );
    }
}

#[test]
fn lookup_matches_an_independent_recursive_reference() {
    let mut random = 0xd4a7_91c3_6ef2_580b_u64;
    for sample in 0..500 {
        random = next_random(random);
        let fixed_melds = random as usize % 5;
        random = next_random(random);
        let missing = match random % 4 {
            0 => Some(Suit::Characters),
            1 => Some(Suit::Bamboo),
            2 => Some(Suit::Dots),
            _ => None,
        };

        let meld_suits: Vec<_> = Suit::ALL
            .into_iter()
            .filter(|&suit| Some(suit) != missing)
            .collect();
        let melds: Vec<_> = (0..fixed_melds)
            .map(|index| meld(meld_suits[index % meld_suits.len()], (index / 2 + 1) as u8))
            .collect();
        let mut limits = [4_u8; 27];
        for exposed in &melds {
            limits[exposed.tile.index()] -= 3;
        }

        let mut counts = [0_u8; 27];
        for _ in 0..14 - 3 * fixed_melds {
            loop {
                random = next_random(random);
                let index = random as usize % counts.len();
                if counts[index] < limits[index] {
                    counts[index] += 1;
                    break;
                }
            }
        }

        assert_eq!(
            evaluate_shanten(&counts, &melds, missing),
            reference_shanten(&counts, fixed_melds, missing),
            "sample={sample}, fixed_melds={fixed_melds}, missing={missing:?}, counts={counts:?}"
        );
    }
}

#[test]
fn batch_analysis_matches_active_games_and_marks_terminal_slots() {
    let mut batch = Batch::new(1, 91);
    let actor = batch.games()[0]
        .decision()
        .expect("a new game has a decision")
        .actor;
    let expected = batch.games()[0].hand_analysis(actor);
    let mut shanten = [0_i8];
    let mut improving_tiles = [0_u32];
    batch
        .hand_analysis_into(&mut shanten, &mut improving_tiles)
        .expect("buffers match the batch");
    assert_eq!(shanten[0], expected.shanten);
    assert_eq!(improving_tiles[0], expected.improving_tiles);
    assert_eq!(
        batch.hand_analysis_into(&mut [], &mut improving_tiles),
        Err(GameError::BatchLength)
    );

    while let Some(action) = batch.games()[0].simple_rule_action() {
        batch.games_mut()[0]
            .step_id(action)
            .expect("the rule policy emits legal actions");
    }
    batch
        .hand_analysis_into(&mut shanten, &mut improving_tiles)
        .expect("buffers match the terminal batch");
    assert_eq!(shanten, [SHANTEN_TERMINAL]);
    assert_eq!(improving_tiles, [0]);
}

fn next_random(mut value: u64) -> u64 {
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

fn reference_shanten(counts: &Counts, fixed_melds: usize, missing: Option<Suit>) -> i8 {
    let mut usable = *counts;
    if let Some(suit) = missing {
        let start = suit as usize * 9;
        usable[start..start + 9].fill(0);
    }

    let mut standard = 8_i8;
    reference_standard_shapes(&mut usable, 0, fixed_melds, 0, false, &mut standard);
    if fixed_melds != 0 {
        return standard;
    }

    let pairs: i8 = usable.iter().map(|&count| (count / 2) as i8).sum();
    let pair_slots: i8 = usable.iter().map(|&count| count.div_ceil(2) as i8).sum();
    let seven_pairs = 6 - pairs + (7 - pair_slots).max(0);
    standard.min(seven_pairs)
}

fn reference_standard_shapes(
    counts: &mut Counts,
    start: usize,
    melds: usize,
    taatsu: usize,
    has_pair: bool,
    best: &mut i8,
) {
    let Some(index) = (start..counts.len()).find(|&index| counts[index] != 0) else {
        let useful_taatsu = taatsu.min(4 - melds);
        *best = (*best).min(8 - 2 * melds as i8 - useful_taatsu as i8 - has_pair as i8);
        return;
    };

    // Leave one tile unused. Repeating this branch also handles duplicate
    // leftovers without baking any lookup-table assumptions into the test.
    counts[index] -= 1;
    reference_standard_shapes(counts, index, melds, taatsu, has_pair, best);
    counts[index] += 1;

    if melds < 4 && counts[index] >= 3 {
        counts[index] -= 3;
        reference_standard_shapes(counts, index, melds + 1, taatsu, has_pair, best);
        counts[index] += 3;
    }
    let rank = index % 9;
    if melds < 4 && rank <= 6 && counts[index + 1] != 0 && counts[index + 2] != 0 {
        counts[index] -= 1;
        counts[index + 1] -= 1;
        counts[index + 2] -= 1;
        reference_standard_shapes(counts, index, melds + 1, taatsu, has_pair, best);
        counts[index] += 1;
        counts[index + 1] += 1;
        counts[index + 2] += 1;
    }
    if !has_pair && counts[index] >= 2 {
        counts[index] -= 2;
        reference_standard_shapes(counts, index, melds, taatsu, true, best);
        counts[index] += 2;
    }
    if taatsu < 4 && counts[index] >= 2 {
        counts[index] -= 2;
        reference_standard_shapes(counts, index, melds, taatsu + 1, has_pair, best);
        counts[index] += 2;
    }
    for gap in 1..=2 {
        if taatsu < 4 && rank + gap < 9 && counts[index + gap] != 0 {
            counts[index] -= 1;
            counts[index + gap] -= 1;
            reference_standard_shapes(counts, index, melds, taatsu + 1, has_pair, best);
            counts[index] += 1;
            counts[index + gap] += 1;
        }
    }
}
