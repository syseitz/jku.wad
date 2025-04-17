# TODO

## Tutors
- [ ] Check if it works in colab
- [ ] Requirements and environment
- [ ] Exercise notebook for students (basic training / model)
- [ ] Model saving/loading from ONNX (check [tournament.ipynb](/tournament.ipynb))
    - [ ] User config in yaml format
    - [ ] decide a configuration format so that each player can set up the a custom env in the tournament (eg custom n frames, different buffers)
    - [ ] ONNX loading in graded / tournament
- [ ] Train a few agents
    - [ ] Basic reward (eg `FRAGS + 0.1 * HITCOUNT - 0.1 * HITS_TAKEN`)
    - [ ] Get baseline reward for grading

## GG
- [ ] Implement graded PvE script
- [x] Implement tournament PvP script
    - [x] Add custom colors and names for characters
    - [x] Include bots and adjust skill https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/cig_multiplayer_bots.py
    - [ ] Live leaderboard on wandb?
    - [x] Plotting function for tournament results
- [x] Create a custom `.wad` / `.cfg` for the tournament (start from cig or multi) - https://doom.fandom.com/wiki/Choosing_a_WAD_editor
    - [x] Also used to make bots slower
    - [x] Explore other maps / import them in `jku.wad`