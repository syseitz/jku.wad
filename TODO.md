# TODO

## Usage
- [ ] Check if it works in colab / prepare a tutorial on how to install it there.
- [ ] Write env docs and tutorial notebook
- [ ] Model saving/loading from ONNX
- [ ] Include game state in observation
- [ ] Provide a few utils from frame / game state preprocessing
- [ ] Train a few agents
    - [ ] Investigate original vizdoom reward
    - [ ] Basic reward
- [ ] Investigate game variables
    - [ ] Check HITCOUNT and DAMAGECOUNT
    - [ ] DEATHCOUNT starts at 1 sometimes


## Tournament
- [x] Create a custom `.wad` / `.cfg` for the tournament (start from cig or multi) - https://doom.fandom.com/wiki/Choosing_a_WAD_editor
    - [x] Also used to make bots slower
    - [ ] Explore other maps / import them in `jku.wad`
- [ ] Check out spectator mode (ASYNC_SPECTATOR)
- [ ] Speed up rendering (eg don't use matplotlib)
- [ ] Implement tournament script
    - [x] Add custom colors and names for characters
    - [x] Include bots and adjust skill https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/cig_multiplayer_bots.py
    - [ ] decide a configuration format so that each player can set up the a custom env in the tournament (eg custom n frames, different buffers)
    - [ ] Live leaderboard on wandb?

## Environment
- [x] Add buffers: labels, automap and depth https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/buffers.py
    - [ ] check what they look like, get snapshots for documentation
- [x] Frame stacking
- [ ] Investigate action space (`adjust_action`)
- [ ] ~~Move transforms outside (apply on batch)~~

## Misc
- [x] Are there heals / armor / weapons / power ups? Only map02
- [ ] ~~Feature extraction network / provide prettained weights~~
- [ ] (optional) Workaround for clearning console prints when multiplayer?
- [ ] (optional) Play around with render options to make game easier? https://vizdoom.farama.org/api/python/doomGame/#output-rendering-setting-methods

