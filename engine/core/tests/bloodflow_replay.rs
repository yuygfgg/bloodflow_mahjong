use bloodflow_mahjong::{ActionId, Game, GameError, Seat};

#[test]
fn historical_winning_tile_cannot_complete_an_added_kong() {
    const SEED: u64 = 11_973_435_932_242_856_896;
    const ACTIONS: &[usize] = &[
        20, 22, 24, 0, 1, 2, 8, 0, 6, 4, 1, 5, 29, 27, 29, 27, 55, 57, 48, 41, 56, 57, 39, 40, 41,
        57, 51, 46, 58, 53, 44, 48, 49, 47, 58, 35, 51, 41, 50, 58, 36, 45, 31, 58, 39, 44, 51, 42,
        30, 32, 58, 30, 40, 37, 48, 56, 57, 55, 57, 31, 57, 30, 44, 42, 88,
    ];

    let mut game = Game::new(SEED);
    let (illegal, prefix) = ACTIONS
        .split_last()
        .expect("the replay contains the invalid added Kong");

    for (step, &action) in prefix.iter().enumerate() {
        let action = ActionId::new(action).expect("the replay action is in range");
        game.step_id(action)
            .unwrap_or_else(|error| panic!("replay action {step} must remain legal: {error}"));
    }

    let illegal = ActionId::new(*illegal).expect("the replay action is in range");
    assert!(game.has_won(Seat::EAST));
    assert_eq!(
        game.concealed(Seat::EAST)[1] - game.locked(Seat::EAST)[1],
        0
    );
    assert!(!game.legal_action_mask().unwrap().contains(illegal));
    assert_eq!(game.step_id(illegal), Err(GameError::InvalidAction));
}

#[test]
fn post_win_active_quad_can_form_a_concealed_kong() {
    const SEED: u64 = 490_937_470_468_900_088;
    const ACTIONS: &[usize] = &[
        9, 11, 15, 17, 13, 11, 0, 6, 4, 23, 19, 24, 29, 28, 27, 29, 53, 58, 31, 48, 55, 58, 39, 34,
        58, 52, 41, 58, 56, 38, 48, 33, 47, 53, 38, 39, 48, 50, 43, 58, 47, 40, 39, 56, 41, 46, 58,
        47, 36, 58, 35, 58, 36, 37, 40, 39, 49, 58, 50, 52, 37, 38, 31, 57, 37, 48, 54, 57, 42, 57,
        52, 35, 52, 56,
    ];

    let mut game = Game::new(SEED);
    for (step, &action) in ACTIONS.iter().enumerate() {
        let action = ActionId::new(action).expect("the replay action is in range");
        game.step_id(action)
            .unwrap_or_else(|error| panic!("replay action {step} must remain legal: {error}"));
    }

    assert!(game.has_won(Seat::EAST));
    assert_eq!(game.concealed(Seat::EAST)[0], 4);
    assert_eq!(game.locked(Seat::EAST)[0], 3);
    assert_eq!(game.public_win_tiles(Seat::EAST)[0], 0);
    let concealed_kong = ActionId::new(60).expect("one-character concealed Kong is in range");
    assert!(
        game.legal_action_mask()
            .expect("the replay stops on a live turn")
            .contains(concealed_kong)
    );
}
