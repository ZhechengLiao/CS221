# MDP

markov decision process
----
Define Problem
1. States
2. Start
3. action
4. chance node 
5. transation probability
6. reward
7. End
8. discount: how long I care about(lvie in the momemnt: 0; save for the future: 1; balanced life: .5)

How to solve
1. policy(map from s -> a)
2. utility: sum of all reward
3. value: expected value of utility

Define the value of a policy V(s)
Define the value of Q(s, a): recommandation for what you should do

policy evaluation - iterative algorithm
```
init V(s) = 0:
    for t in 1 ... tpe:
        for each state:
            V(s) = sum of T*[Reword + discount*V(s')]
```
Big-O: O(tpe*s*s')

policy optimization: pick the highest value(max)
```
init V(s) = 0:
    for t in 1 ... t:
        for each state:
            V(s) = max(sum of T*[Reward + discount*V(s')])
```


Reinforcement Learning
----
MDP(offline) -> Reinforcement Learning(online)

```
for t = 1 ... n:
    agent -> (action) -> environment
    evironment -> (reward) -> agent
    update parameters
```

Model Based Value Iteration
----
Need to learn the parameters
Define T, and reward
- T = #(s, a, s') / #(s, a)
- reward

Model Free monte carlo
----
No need to learn
Estimate: Q(s, a)
