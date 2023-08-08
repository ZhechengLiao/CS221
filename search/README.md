# Search
> HW 3 is out

**States model**, express state
- Route finding
- motion planing
- solving puzzles
- machine translation

**output -> sequence of actions(a, b, c, d, ...)**

x -> f -> action sequence, consider the future consequences of the action

**Inference** is important

## Modeling
- what state I'm in
tips:
    - start state.
    - action: possible actions.
    - cost: cost action cost, walk from to one state to another state, there is trade off.
    - succ: successor.
    - IsEnd: reched end state.
## Tree Search
### Backtracking Search
traversal all the nodes, we can get the shortest way, but its cost tooooo much.

### DFS(Depth First Search)
- cost = 0, cost doesn't make sense. (we don't care about the cost, but we have cost)
- backtracking + stop when find the first end state.

### BFS(Breadth First Search)
- cost = c, c >= 0
- explore all nodes in order of increasing depth. (like drop water into box)

### DFS with iterative deepening
- call DFS for maximum depth 1, 2, 3, ...
- cost = c, c >= 0
- look at different depth, whether we can find in this depth. 

### Summary Table
|Name|Space|Time|
|----|----|----|
|Backtracking Search|O(D)(small)|O(b^D)(huge)|
|DFS|O(D)|O(b^D) - could be better in reality|
|BFS|O(b^d)|O(b^d)|
|DFS-ID|O(d)|O(b^d)|

## Dynamic Programming
state1 -> cost -> state2 -> **future cost** -> end state
$futureCost(s) = min(a)|Cost(s, a) + FutureCost(succ(s, a))|$
same sub question -> dynamic program

- Recursive
- Store(memorize)
- bottom-up

```
def dynamic program:
    if isEnd: return solution
    for each actionm: ...
```
### state
- a state summary of all past actions sufficeint to choose future actions optimally
- Constrains
  - add constrains to upgrade state

## Uniform cost search
- start state -> past cost(s) -> s -> cost(s, a) -> s'
- if a graph is acyclic, dynamic programming makes sure we compute past cost(s) before past cost(s').
**UCS enumerates states in order of increasing past cost**

- Explored: states we found the optimal path to
- Frontier: states still figureing out how to get there cheaply
- Unexplored:states we haven't seen

```
add start to frontier
repeat utill frontier is empty:
    Remove
```
## A*
Idea: explore in order of pastcost + $h(s)$
$cost(s, a)' = cost(s, a) + [h(s') - h(s)]$, where $h(s)$ can help me predict cost for future cost.
$h(x)$ is heuristic, there is method to find a good $h(x)$.
- cost'(s, a) >= 0
- h(s_end) = 0

### How to find $h(x)$ - A* Relaxations
Relaxations: Remove **Constrains**, make the problem easier.


# Markov Decision process
Search: state, action -> state succ

**Uncertainty in the real world**

 







