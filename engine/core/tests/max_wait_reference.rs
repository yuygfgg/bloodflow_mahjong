use bloodflow_mahjong::{
    MaxWaitEvaluation, Meld, MeldKind, Pattern, Seat, Suit, Tile, WinFlags, evaluate_max_wait,
    evaluate_win,
};

type Counts = [u8; 27];

const EVENT_PATTERNS: [Pattern; 6] = [
    Pattern::RobKong,
    Pattern::KongDiscard,
    Pattern::KongDraw,
    Pattern::LastWallTile,
    Pattern::Heavenly,
    Pattern::Earthly,
];

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

fn meld(suit: Suit, rank: u8, kind: MeldKind) -> Meld {
    Meld {
        tile: tile(suit, rank),
        kind,
        source: Seat::EAST,
    }
}

fn assert_structural_only(wait: &MaxWaitEvaluation) {
    assert_eq!(
        wait.evaluation.multiplier, wait.evaluation.shape_multiplier,
        "dajiao must not include an event multiplier"
    );
    for event in EVENT_PATTERNS {
        assert!(
            !wait.evaluation.patterns.contains(event),
            "dajiao unexpectedly included {event:?}"
        );
    }
}

fn assert_max_wait(
    counts: &Counts,
    melds: &[Meld],
    missing_suit: Option<Suit>,
    expected_tile: Tile,
    expected_multiplier: u32,
) -> MaxWaitEvaluation {
    let wait = evaluate_max_wait(counts, melds, missing_suit).expect("expected a ready hand");
    assert_eq!(wait.winning_tile, expected_tile);
    assert_eq!(wait.evaluation.shape_multiplier, expected_multiplier);
    assert_eq!(wait.evaluation.multiplier, expected_multiplier);
    assert_structural_only(&wait);
    wait
}

fn exact_standard_hand(counts: &Counts) -> bool {
    if counts
        .iter()
        .map(|&count| usize::from(count))
        .sum::<usize>()
        != 14
    {
        return false;
    }

    let mut remaining = *counts;
    for pair in 0..remaining.len() {
        if remaining[pair] < 2 {
            continue;
        }
        remaining[pair] -= 2;
        if exact_melds(&mut remaining, 4) {
            return true;
        }
        remaining[pair] += 2;
    }
    false
}

fn exact_melds(counts: &mut Counts, remaining: usize) -> bool {
    let Some(first) = counts.iter().position(|&count| count != 0) else {
        return remaining == 0;
    };
    if remaining == 0 {
        return false;
    }

    if counts[first] >= 3 {
        counts[first] -= 3;
        if exact_melds(counts, remaining - 1) {
            counts[first] += 3;
            return true;
        }
        counts[first] += 3;
    }

    let rank = first % 9;
    if rank <= 6 && counts[first + 1] != 0 && counts[first + 2] != 0 {
        counts[first] -= 1;
        counts[first + 1] -= 1;
        counts[first + 2] -= 1;
        if exact_melds(counts, remaining - 1) {
            counts[first] += 1;
            counts[first + 1] += 1;
            counts[first + 2] += 1;
            return true;
        }
        counts[first] += 1;
        counts[first + 1] += 1;
        counts[first + 2] += 1;
    }
    false
}

#[test]
fn multi_wait_search_uses_the_highest_scoring_tile() {
    // 1s makes every group contain a terminal (4x); 4s is only a plain win (1x).
    let counts = hand(&[
        (Suit::Characters, 1, 1),
        (Suit::Characters, 2, 1),
        (Suit::Characters, 3, 1),
        (Suit::Characters, 7, 1),
        (Suit::Characters, 8, 1),
        (Suit::Characters, 9, 1),
        (Suit::Bamboo, 1, 3),
        (Suit::Bamboo, 2, 1),
        (Suit::Bamboo, 3, 1),
        (Suit::Dots, 9, 2),
    ]);

    let low_tile = tile(Suit::Bamboo, 4);
    let mut low_counts = counts;
    low_counts[low_tile.index()] += 1;
    let low = evaluate_win(&low_counts, &[], Some(low_tile), WinFlags::NONE)
        .expect("4s is the lower-scoring side of the wait");
    assert_eq!(low.multiplier, 1);

    let wait = assert_max_wait(
        &counts,
        &[],
        None,
        tile(Suit::Bamboo, 1),
        Pattern::TerminalsInEveryGroup.multiplier(),
    );
    assert!(
        wait.evaluation
            .patterns
            .contains(Pattern::TerminalsInEveryGroup)
    );
}

#[test]
fn seven_pairs_beats_a_standard_decomposition_of_the_same_wait() {
    // 11223344556677m is both 123/123/456/456/77 and seven pairs.
    let mut counts = [0; 27];
    for rank in 1..=6 {
        counts[tile(Suit::Characters, rank).index()] = 2;
    }
    counts[tile(Suit::Characters, 7).index()] = 1;

    let winning_tile = tile(Suit::Characters, 7);
    let mut completed = counts;
    completed[winning_tile.index()] += 1;
    assert!(
        exact_standard_hand(&completed),
        "the reference decomposition must really be ambiguous"
    );

    let wait = assert_max_wait(&counts, &[], None, winning_tile, 16);
    assert!(wait.evaluation.patterns.contains(Pattern::PureSevenPairs));
    assert!(!wait.evaluation.patterns.contains(Pattern::PureOneSuit));
}

#[test]
fn four_exposed_groups_are_scored_as_golden_hook() {
    let melds = [
        meld(Suit::Characters, 1, MeldKind::Pong),
        meld(Suit::Bamboo, 3, MeldKind::ExposedKong),
        meld(Suit::Dots, 5, MeldKind::AddedKong),
        meld(Suit::Dots, 7, MeldKind::ConcealedKong),
    ];
    let pair_tile = tile(Suit::Characters, 5);
    let counts = hand(&[(Suit::Characters, 5, 1)]);

    let wait = assert_max_wait(&counts, &melds, None, pair_tile, 4);
    assert!(wait.evaluation.patterns.contains(Pattern::GoldenHook));
    assert!(!wait.evaluation.patterns.contains(Pattern::EighteenArhats));
}

#[test]
fn exposed_melds_cannot_use_a_seven_pairs_substructure() {
    let melds = [meld(Suit::Characters, 1, MeldKind::Pong)];
    let counts = hand(&[
        (Suit::Characters, 2, 2),
        (Suit::Characters, 5, 2),
        (Suit::Characters, 8, 2),
        (Suit::Bamboo, 2, 2),
        (Suit::Bamboo, 5, 2),
        (Suit::Bamboo, 8, 2),
        (Suit::Dots, 5, 2),
    ]);

    assert_eq!(
        evaluate_max_wait(&counts, &melds, None),
        None,
        "seven pairs is invalid after any pong or kong"
    );
}

#[test]
fn a_missing_suit_in_concealed_or_exposed_tiles_excludes_the_hand() {
    // Without dingque this expanded hand can ignore 1p and win on 9s.
    let concealed_missing = hand(&[
        (Suit::Characters, 1, 1),
        (Suit::Characters, 2, 1),
        (Suit::Characters, 3, 1),
        (Suit::Characters, 4, 1),
        (Suit::Characters, 5, 1),
        (Suit::Characters, 6, 1),
        (Suit::Bamboo, 1, 1),
        (Suit::Bamboo, 2, 1),
        (Suit::Bamboo, 3, 1),
        (Suit::Bamboo, 4, 1),
        (Suit::Bamboo, 5, 1),
        (Suit::Bamboo, 6, 1),
        (Suit::Bamboo, 9, 1),
        (Suit::Dots, 1, 1),
    ]);
    assert!(evaluate_max_wait(&concealed_missing, &[], None).is_some());
    assert_eq!(
        evaluate_max_wait(&concealed_missing, &[], Some(Suit::Dots)),
        None
    );

    let exposed_missing = [
        meld(Suit::Characters, 1, MeldKind::Pong),
        meld(Suit::Characters, 3, MeldKind::Pong),
        meld(Suit::Bamboo, 5, MeldKind::Pong),
        meld(Suit::Dots, 7, MeldKind::Pong),
    ];
    let single = hand(&[(Suit::Characters, 5, 1)]);
    assert!(evaluate_max_wait(&single, &exposed_missing, None).is_some());
    assert_eq!(
        evaluate_max_wait(&single, &exposed_missing, Some(Suit::Dots)),
        None
    );
}

#[test]
fn logical_duplicate_counts_above_four_still_find_the_best_wait() {
    let counts = hand(&[
        (Suit::Characters, 1, 9),
        (Suit::Characters, 2, 3),
        (Suit::Bamboo, 3, 3),
        (Suit::Dots, 4, 3),
        (Suit::Dots, 5, 1),
    ]);

    let wait = assert_max_wait(&counts, &[], None, tile(Suit::Dots, 5), 2);
    assert!(wait.evaluation.patterns.contains(Pattern::AllTriplets));
    assert_eq!(wait.evaluation.used[tile(Suit::Characters, 1).index()], 3);
    assert!(wait.evaluation.used.iter().all(|&count| count <= 4));
}

#[test]
fn unrelated_tiles_have_no_wait() {
    let counts = hand(&[
        (Suit::Characters, 1, 1),
        (Suit::Characters, 3, 1),
        (Suit::Characters, 5, 1),
        (Suit::Characters, 7, 1),
        (Suit::Characters, 9, 1),
        (Suit::Bamboo, 1, 1),
        (Suit::Bamboo, 3, 1),
        (Suit::Bamboo, 5, 1),
        (Suit::Bamboo, 7, 1),
        (Suit::Bamboo, 9, 1),
        (Suit::Dots, 1, 1),
        (Suit::Dots, 5, 1),
        (Suit::Dots, 9, 1),
    ]);

    assert_eq!(evaluate_max_wait(&counts, &[], None), None);
}
