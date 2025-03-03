# TODO

## Tournament
- [ ] Implement tournament script
    - [x] Add custom colors and names for characters
    - [x] Add non-ml bots https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/cig_multiplayer_bots.py
    - [ ] decide a configuration format so that each player can set up the a custom env in the tournament (eg custom n frames, different buffers)

## Environment
- [x] Add buffers: labels, automap and depth https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/buffers.py
- [x] Frame stacking
- [ ] Move transforms outside (apply on batch)

## Misc
- [ ] Basic reward / train DQN
- [ ] Check if game variables are updated (HITCOUNT)
- [x] Are there heals / armor / weapons / power ups? Only map02
- [ ] ~~Feature extraction network / provide prettained weights~~

