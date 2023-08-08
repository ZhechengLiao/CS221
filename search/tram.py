class TramProblem:
    def __init__(self, N):
        # N = number of blocks
        self.N = N

    def start(self):
        return 1
    
    def isEnd(self, state):
        return state == self.N

    def succAndCost(self, state):
        # return list of (action, newState, cost) triples
        result = []
        if state+1 <= self.N:
            result.append(('walk', state+1, 1))
        if state*2 <= self.N:
            result.append(('magic', state*2, 2))

        return result

def printSolution(solution):
    totalCost, history = solution
    print('totalCost:{}'.format(totalCost))
    for item in history:
        print(item)

def backtrackingSearch(problem):
    best = {
        'cost': 10000000,
        'history': None
    }
    def recurse(state, history, totalCost):
        if problem.isEnd(state):
          # update new state info
            if totalCost < best['cost']:
              best['cost'] = totalCost
              best['history'] = history
        
        # recurse the state
        for action, newState, cost in problem.succAndCost(state):
            recurse(newState, history+[(action, newState, cost)], totalCost+cost)
    
    recurse(problem.start(), history=[], totalCost=0)

    return (best['cost'], best['history'])

def dynamicProgramming(problem):
    def futureCost(state):
        # base
        if problem.isEnd(state):
            return 0
        # dynamic programming
        result = min(cost+futureCost(newState) for action, newState, cost in problem.succAndCost(state))
        return result
    
    return (futureCost(problem.start()), [])
    

problem1 = TramProblem(N=30)
print(problem1.succAndCost(4))
# printSolution(backtrackingSearch(problem1))
# printSolution(dynamicProgramming(problem1))

