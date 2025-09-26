# InternNav Core Modules

## agent
```
class Agent:
    step: generate action for one step in simulation world
    reset: reset state to start another task episode
```

## model
```
class Model:
    forward: return (loss, pred)
    inference: make prediction
```
## evaluator
```
class Evaluator:
    # gym style evaluation loop
    eval: main evaluation
```
## env
```
class Env:
    step:
    reset:
    close:
    render:
    get_observation:
```
