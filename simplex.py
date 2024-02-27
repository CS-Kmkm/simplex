import numpy as np

class SimplexProblem(object):
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c
        if not isinstance(self, SubSimplexProblem):
            self.initial_answer()
            self.idx_base = []
            self.idx_non_base = []
            for i in range(len(self.c)):
                if self.x[i] != 0:
                    self.idx_base.append(i)
                else:
                    self.idx_non_base.append(i)

    def initial_answer(self):
        self.subproblem = SubSimplexProblem(self.A, self.b, self.c)
        self.subproblem.solve()
        if np.any(self.subproblem.c.T @ self.subproblem.x > 0):
            print("The problem is not feasible")
            exit()
        self.x = self.subproblem.x[:len(self.c)]

    def solve(self):
        while True:
            if self.solve_iteration():
                break

    def solve_iteration(self):
        simplex_multipliers = self.calc_simplex_multipliers()
        relative_cost_coefficients = self.calc_relative_cost_coefficients(simplex_multipliers)
        if np.all(relative_cost_coefficients >= 0):
            print("The problem is optimal")
            print(self.x)
            return True
        k = self.idx_non_base[np.argmin(relative_cost_coefficients)]
        y = self.calc_y(k)
        if np.all(y <= 0):
            print("The problem is unbounded")
            exit()
        args, thetas = self.calc_theta(y)
        theta = min(thetas)
        idx = args[thetas.index(theta)]

        self.pivot(k, idx, theta, y)

    def calc_simplex_multipliers(self):
        B = self.A[:, self.idx_base]
        return np.linalg.inv(B.T) @ self.c[self.idx_base]

    def calc_relative_cost_coefficients(self, simplex_multipliers):
        N = self.A[:, self.idx_non_base]
        return self.c[self.idx_non_base] - N.T @ simplex_multipliers
    
    def calc_y(self, idx):
        B = self.A[:, self.idx_base]
        return np.linalg.inv(B) @ self.A[:, idx]

    def calc_theta(self, y):
        args = []
        thetas = []
        B = self.A[:, self.idx_base]
        b = np.linalg.inv(B) @ self.b

        for i in range(len(self.b)):
            if y[i] > 0:
                args.append(self.idx_base[i])
                thetas.append(b[i] / y[i])
        return args, thetas
    
    def pivot(self, k, idx, theta, y):
        B = self.A[:, self.idx_base]
        b_bar = np.linalg.inv(B) @ self.b
        self.idx_base[self.idx_base.index(idx)] = k

        for i, b in enumerate(self.idx_base):
            self.x[b] = b_bar[i] - theta * y[i]

        self.idx_non_base[self.idx_non_base.index(k)] = idx
        for nb in self.idx_non_base:
            self.x[nb] = 0

        self.x[k] = theta

        self.idx_base.sort()
        self.idx_non_base.sort()

class SubSimplexProblem(SimplexProblem):
    def __init__(self, A, b, c):
        super().__init__(A, b, c)
        self.idx_base = []
        self.idx_non_base = list(range(len(self.c)))

        for i in range(len(self.b)):
            self.idx_base.append(len(self.c) + i)

        for i, each_b in enumerate(b):
            vector = np.zeros(len(b))
            if each_b < 0:
                vector[i] = -1
            else:
                vector[i] = 1
            vector = vector.reshape(len(b), 1)
            self.A = np.append(self.A, vector, axis=1)
        self.c = np.concatenate([np.zeros(len(self.c)), np.ones(len(self.b))], 0)
        
        self.x = np.zeros(len(self.A[0]))

if __name__ == "__main__":
    A = np.array([[3, 2, 1], [1, 2, 4]])
    b = np.array([12, 8])
    c = np.array([-1, -1, 0])
    problem = SimplexProblem(A, b, c)
    problem.solve()