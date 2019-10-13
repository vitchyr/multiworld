# roboverse

## Setup
`pip install -r requirements.txt`

## Quick start
```python
from roboverse.envs.sawyer_reach import SawyerReachEnv
env = SawyerReachEnv(renders=True)
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
```

### TODO
- [ ] Add vision API (cameras, observation dictionaries etc.)
- [ ] Add keyboard control script
- [ ] Add more environments - pushing, grasping
- [ ] Clean up code to avoid code repetition when multiple envs are added
- [ ] Add infrastructure for loading a large number of different objects from ShapeNet