use bloodflow_mahjong::{Meld, MeldKind, Pattern, Seat, Suit, Tile, WinFlags, evaluate_win};

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

fn meld(suit: Suit, rank: u8, kind: MeldKind) -> Meld {
    Meld {
        tile: tile(suit, rank),
        kind,
        source: Seat::EAST,
    }
}

fn assert_score(
    label: &str,
    counts: &Counts,
    melds: &[Meld],
    flags: WinFlags,
    expected_shape: u32,
    expected_total: u32,
    expected_patterns: &[Pattern],
) {
    let evaluation = evaluate_win(counts, melds, None, flags)
        .unwrap_or_else(|| panic!("{label}: expected a winning hand"));
    let actual_patterns: Vec<_> = evaluation.patterns.iter().collect();

    assert_eq!(
        evaluation.shape_multiplier, expected_shape,
        "{label}: wrong structural multiplier"
    );
    assert_eq!(
        evaluation.multiplier, expected_total,
        "{label}: wrong total multiplier"
    );
    assert_eq!(
        actual_patterns, expected_patterns,
        "{label}: wrong pattern selection or suppression"
    );
}

fn plain_hand() -> Counts {
    hand(&[
        (Suit::Characters, 1, 1),
        (Suit::Characters, 2, 1),
        (Suit::Characters, 3, 1),
        (Suit::Bamboo, 4, 1),
        (Suit::Bamboo, 5, 1),
        (Suit::Bamboo, 6, 1),
        (Suit::Dots, 7, 1),
        (Suit::Dots, 8, 1),
        (Suit::Dots, 9, 1),
        (Suit::Characters, 5, 3),
        (Suit::Bamboo, 9, 2),
    ])
}

#[test]
fn standard_shape_patterns_have_exact_multipliers() {
    assert_score(
        "plain",
        &plain_hand(),
        &[],
        WinFlags::NONE,
        1,
        1,
        &[Pattern::Plain],
    );

    let all_simples = hand(&[
        (Suit::Characters, 2, 1),
        (Suit::Characters, 3, 1),
        (Suit::Characters, 4, 1),
        (Suit::Bamboo, 3, 1),
        (Suit::Bamboo, 4, 1),
        (Suit::Bamboo, 5, 1),
        (Suit::Dots, 4, 1),
        (Suit::Dots, 5, 1),
        (Suit::Dots, 6, 1),
        (Suit::Characters, 6, 1),
        (Suit::Characters, 7, 1),
        (Suit::Characters, 8, 1),
        (Suit::Bamboo, 5, 2),
    ]);
    assert_score(
        "all simples",
        &all_simples,
        &[],
        WinFlags::NONE,
        2,
        2,
        &[Pattern::AllSimples],
    );

    let all_triplets = hand(&[
        (Suit::Characters, 1, 3),
        (Suit::Characters, 5, 3),
        (Suit::Bamboo, 2, 3),
        (Suit::Dots, 9, 3),
        (Suit::Dots, 4, 2),
    ]);
    assert_score(
        "all triplets",
        &all_triplets,
        &[],
        WinFlags::NONE,
        2,
        2,
        &[Pattern::AllTriplets],
    );

    let pure_all_triplets = hand(&[
        (Suit::Characters, 1, 3),
        (Suit::Characters, 3, 3),
        (Suit::Characters, 5, 3),
        (Suit::Characters, 7, 3),
        (Suit::Characters, 9, 2),
    ]);
    assert_score(
        "pure all triplets",
        &pure_all_triplets,
        &[],
        WinFlags::NONE,
        8,
        8,
        &[Pattern::PureAllTriplets],
    );

    let pure_one_suit = hand(&[
        (Suit::Characters, 1, 1),
        (Suit::Characters, 2, 2),
        (Suit::Characters, 3, 2),
        (Suit::Characters, 4, 2),
        (Suit::Characters, 5, 1),
        (Suit::Characters, 6, 2),
        (Suit::Characters, 7, 1),
        (Suit::Characters, 8, 1),
        (Suit::Characters, 9, 2),
    ]);
    assert_score(
        "pure one suit",
        &pure_one_suit,
        &[],
        WinFlags::NONE,
        4,
        4,
        &[Pattern::PureOneSuit],
    );
}

#[test]
fn seven_pairs_family_selects_only_the_highest_specific_variant() {
    let cases: [(&str, Counts, u32, Pattern); 9] = [
        (
            "seven pairs",
            hand(&[
                (Suit::Characters, 1, 2),
                (Suit::Characters, 4, 2),
                (Suit::Characters, 6, 2),
                (Suit::Bamboo, 2, 2),
                (Suit::Bamboo, 7, 2),
                (Suit::Dots, 3, 2),
                (Suit::Dots, 9, 2),
            ]),
            4,
            Pattern::SevenPairs,
        ),
        (
            "pure seven pairs",
            hand(&[
                (Suit::Characters, 1, 2),
                (Suit::Characters, 2, 2),
                (Suit::Characters, 4, 2),
                (Suit::Characters, 5, 2),
                (Suit::Characters, 7, 2),
                (Suit::Characters, 8, 2),
                (Suit::Characters, 9, 2),
            ]),
            16,
            Pattern::PureSevenPairs,
        ),
        (
            "two-five-eight seven pairs",
            hand(&[
                (Suit::Characters, 2, 2),
                (Suit::Characters, 5, 2),
                (Suit::Characters, 8, 2),
                (Suit::Bamboo, 2, 2),
                (Suit::Bamboo, 5, 2),
                (Suit::Dots, 2, 2),
                (Suit::Dots, 8, 2),
            ]),
            16,
            Pattern::TwoFiveEightSevenPairs,
        ),
        (
            "dragon seven pairs",
            hand(&[
                (Suit::Characters, 1, 4),
                (Suit::Characters, 4, 2),
                (Suit::Bamboo, 3, 2),
                (Suit::Bamboo, 6, 2),
                (Suit::Dots, 2, 2),
                (Suit::Dots, 9, 2),
            ]),
            8,
            Pattern::DragonSevenPairs,
        ),
        (
            "pure dragon seven pairs",
            hand(&[
                (Suit::Characters, 1, 4),
                (Suit::Characters, 2, 2),
                (Suit::Characters, 4, 2),
                (Suit::Characters, 5, 2),
                (Suit::Characters, 7, 2),
                (Suit::Characters, 9, 2),
            ]),
            32,
            Pattern::PureDragonSevenPairs,
        ),
        (
            "double dragon seven pairs",
            hand(&[
                (Suit::Characters, 1, 4),
                (Suit::Bamboo, 4, 4),
                (Suit::Characters, 6, 2),
                (Suit::Dots, 3, 2),
                (Suit::Dots, 9, 2),
            ]),
            16,
            Pattern::DoubleDragonSevenPairs,
        ),
        (
            "triple dragon seven pairs",
            hand(&[
                (Suit::Characters, 1, 4),
                (Suit::Bamboo, 4, 4),
                (Suit::Dots, 7, 4),
                (Suit::Characters, 9, 2),
            ]),
            32,
            Pattern::TripleDragonSevenPairs,
        ),
        (
            "two-five-eight double dragon seven pairs",
            hand(&[
                (Suit::Characters, 2, 4),
                (Suit::Bamboo, 5, 4),
                (Suit::Dots, 8, 2),
                (Suit::Characters, 5, 2),
                (Suit::Dots, 2, 2),
            ]),
            64,
            Pattern::TwoFiveEightDoubleDragonSevenPairs,
        ),
        (
            "two-five-eight triple dragon seven pairs",
            hand(&[
                (Suit::Characters, 2, 4),
                (Suit::Bamboo, 5, 4),
                (Suit::Dots, 8, 4),
                (Suit::Bamboo, 2, 2),
            ]),
            128,
            Pattern::TwoFiveEightTripleDragonSevenPairs,
        ),
    ];

    for (label, counts, multiplier, pattern) in cases {
        assert_score(
            label,
            &counts,
            &[],
            WinFlags::NONE,
            multiplier,
            multiplier,
            &[pattern],
        );
    }
}

#[test]
fn terminal_shape_family_has_exact_non_stacking_behavior() {
    let terminals_in_every_group = hand(&[
        (Suit::Characters, 1, 1),
        (Suit::Characters, 2, 1),
        (Suit::Characters, 3, 1),
        (Suit::Characters, 7, 1),
        (Suit::Characters, 8, 1),
        (Suit::Characters, 9, 1),
        (Suit::Bamboo, 1, 3),
        (Suit::Bamboo, 9, 3),
        (Suit::Dots, 1, 2),
    ]);
    assert_score(
        "terminals in every group",
        &terminals_in_every_group,
        &[],
        WinFlags::NONE,
        4,
        4,
        &[Pattern::TerminalsInEveryGroup],
    );

    let all_terminals = hand(&[
        (Suit::Characters, 1, 3),
        (Suit::Characters, 9, 3),
        (Suit::Bamboo, 1, 3),
        (Suit::Bamboo, 9, 3),
        (Suit::Dots, 1, 2),
    ]);
    assert_score(
        "all terminals",
        &all_terminals,
        &[],
        WinFlags::NONE,
        16,
        16,
        &[Pattern::AllTerminals],
    );
}

#[test]
fn independent_structural_card_types_compete_instead_of_multiplying() {
    let pure_all_simples = hand(&[
        (Suit::Characters, 2, 1),
        (Suit::Characters, 3, 2),
        (Suit::Characters, 4, 3),
        (Suit::Characters, 5, 4),
        (Suit::Characters, 6, 2),
        (Suit::Characters, 7, 1),
        (Suit::Characters, 8, 1),
    ]);
    assert_score(
        "pure all simples",
        &pure_all_simples,
        &[],
        WinFlags::NONE,
        4,
        4,
        &[Pattern::AllSimples, Pattern::PureOneSuit],
    );

    let all_simples_seven_pairs = hand(&[
        (Suit::Characters, 2, 2),
        (Suit::Characters, 3, 2),
        (Suit::Characters, 4, 2),
        (Suit::Bamboo, 3, 2),
        (Suit::Bamboo, 4, 2),
        (Suit::Dots, 6, 2),
        (Suit::Dots, 7, 2),
    ]);
    assert_score(
        "all simples seven pairs",
        &all_simples_seven_pairs,
        &[],
        WinFlags::NONE,
        4,
        4,
        &[Pattern::AllSimples, Pattern::SevenPairs],
    );
}

#[test]
fn four_exposed_groups_select_arhats_or_golden_hook_without_base_patterns() {
    let mixed_groups = [
        meld(Suit::Characters, 1, MeldKind::ExposedKong),
        meld(Suit::Characters, 4, MeldKind::AddedKong),
        meld(Suit::Bamboo, 6, MeldKind::ConcealedKong),
        meld(Suit::Dots, 8, MeldKind::ExposedKong),
    ];
    let mixed_pair = hand(&[(Suit::Bamboo, 9, 2)]);
    assert_score(
        "eighteen arhats",
        &mixed_pair,
        &mixed_groups,
        WinFlags::NONE,
        64,
        64,
        &[Pattern::EighteenArhats],
    );

    let pure_groups = [
        meld(Suit::Characters, 1, MeldKind::ExposedKong),
        meld(Suit::Characters, 3, MeldKind::AddedKong),
        meld(Suit::Characters, 5, MeldKind::ConcealedKong),
        meld(Suit::Characters, 7, MeldKind::ExposedKong),
    ];
    let pure_pair = hand(&[(Suit::Characters, 9, 2)]);
    assert_score(
        "pure eighteen arhats",
        &pure_pair,
        &pure_groups,
        WinFlags::NONE,
        256,
        256,
        &[Pattern::PureEighteenArhats],
    );

    let mixed_hook_groups = [
        meld(Suit::Characters, 1, MeldKind::Pong),
        meld(Suit::Characters, 4, MeldKind::AddedKong),
        meld(Suit::Bamboo, 6, MeldKind::ConcealedKong),
        meld(Suit::Dots, 8, MeldKind::ExposedKong),
    ];
    assert_score(
        "golden hook",
        &mixed_pair,
        &mixed_hook_groups,
        WinFlags::NONE,
        4,
        4,
        &[Pattern::GoldenHook],
    );

    let pure_hook_groups = [
        meld(Suit::Characters, 1, MeldKind::Pong),
        meld(Suit::Characters, 3, MeldKind::AddedKong),
        meld(Suit::Characters, 5, MeldKind::ConcealedKong),
        meld(Suit::Characters, 7, MeldKind::ExposedKong),
    ];
    assert_score(
        "pure golden hook",
        &pure_pair,
        &pure_hook_groups,
        WinFlags::NONE,
        16,
        16,
        &[Pattern::PureGoldenHook],
    );
}

#[test]
fn win_event_card_types_compete_with_the_shape_multiplier() {
    let cases = [
        (
            "rob kong",
            WinFlags {
                rob_kong: true,
                ..WinFlags::NONE
            },
            2,
            Pattern::RobKong,
        ),
        (
            "kong discard",
            WinFlags {
                after_kong_discard: true,
                ..WinFlags::NONE
            },
            2,
            Pattern::KongDiscard,
        ),
        (
            "kong draw",
            WinFlags {
                after_kong_draw: true,
                ..WinFlags::NONE
            },
            2,
            Pattern::KongDraw,
        ),
        (
            "last wall tile",
            WinFlags {
                last_wall_tile: true,
                ..WinFlags::NONE
            },
            2,
            Pattern::LastWallTile,
        ),
        (
            "heavenly",
            WinFlags {
                heavenly: true,
                ..WinFlags::NONE
            },
            32,
            Pattern::Heavenly,
        ),
        (
            "earthly",
            WinFlags {
                earthly: true,
                ..WinFlags::NONE
            },
            32,
            Pattern::Earthly,
        ),
    ];

    let counts = plain_hand();
    for (label, flags, event_multiplier, pattern) in cases {
        assert_score(
            label,
            &counts,
            &[],
            flags,
            1,
            event_multiplier,
            &[Pattern::Plain, pattern],
        );
    }

    assert_score(
        "kong draw on the last wall tile",
        &counts,
        &[],
        WinFlags {
            after_kong_draw: true,
            last_wall_tile: true,
            ..WinFlags::NONE
        },
        1,
        2,
        &[Pattern::Plain, Pattern::KongDraw, Pattern::LastWallTile],
    );
}
