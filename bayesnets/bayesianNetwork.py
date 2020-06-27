import numpy as np
import random
import copy


class Factor:
    def __init__(self, prob: np.ndarray, param=tuple()):
        self.prob = prob
        self.param = {item: i for i, item in enumerate(param)}

    def contain(self, var: str) -> bool:
        return var in self.param.keys()

    def select(self, evidence: dict) -> None:
        result = self
        slices = list()

        for x in self.param:
            try:
                idx = evidence[x]
                slices.append(slice(idx, idx + 1))
            except KeyError:
                slices.append(slice(None))

        self.prob = self.prob[tuple(slices)]

    def reduce(self, var: str) -> None:
        try:
            idx = self.param.pop(var)
            self.prob = self.prob.sum(axis=idx)
            self.param = {item: i for i, item in enumerate(
                list(self.param.keys()))}
        except:
            pass

    def product(self, Y: 'Factor' = None) -> 'Factor':
        if Y == None:
            return self

        X = self
        # new [arameter of factor product
        label_z = X.param.keys() & Y.param.keys()
        label_x = [x for x in X.param if x not in label_z]
        label_y = [y for y in Y.param if y not in label_z]
        param = [*label_x, *label_z, *label_y]

        # conver parameter to numpy index
        idx_x = [X.param[item] for item in label_x]
        idx_y = [Y.param[item] for item in label_y]
        idx_ins_x = [X.param[x] for x in label_z]
        idx_ins_y = [Y.param[y] for y in label_z]

        # reshape factor conditional distribution
        X_t = np.transpose(X.prob, axes=tuple(idx_x + idx_ins_x))
        X_t = X_t.reshape(*X_t.shape, *([1] * (len(Y.param) - len(label_z))))
        Y_t = np.transpose(Y.prob, axes=tuple(idx_ins_y + idx_y))

        # product 2 factors
        prob = X_t * Y_t

        return Factor(prob, param)

    def __str__(self):
        return str({"param": list(self.param.keys()), "shape": self.prob.shape})


class BayesianNetwork:
    def __init__(self, filename):
        self.network = {}
        f = open(filename, 'r')
        N = int(f.readline())
        lines = f.readlines()

        self.nodes = []
        self.factors = []
        self.dom2idx = {}

        for line in lines:
            node, parents, domain, shape, prob = extract_model(line)
            ls_dom_idx = list(range(len(domain)))

            self.nodes.append(node)
            self.factors.append(Factor(prob, parents + [node]))
            self.dom2idx[node] = dict(zip(domain, ls_dom_idx))

        f.close()

    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_var, evid_var = self.__extract_query(f.readline())

        ls_factor = copy.deepcopy(self.factors)

        # Check exceoption
        for x in evid_var:
            try:
                if evid_var[x] == query_var[x]:
                    query_var.pop(x)
                else:
                    return 0
            except:
                pass

        # Step 1: select probability factor
        for F in ls_factor:
            F.select(evid_var)

        # Step 2: Find hiden variable set (in-order)
        hiden_vars = list()
        for x in self.nodes:
            if x not in query_var:
                if x not in evid_var:
                    hiden_vars.append(x)

        # Step 3: Remove hiden variables
        for z in hiden_vars:
            new_factor = None
            for F in ls_factor:
                if F.contain(z):
                    new_factor = F.product(new_factor)

            filtered = filter(lambda x: not x.contain(z), ls_factor)
            ls_factor = list(filtered)

            if new_factor != None:
                new_factor.reduce(z)
                ls_factor.append(new_factor)

        # Step 4: Product Fator
        prod = None
        for F in ls_factor:
            prod = F.product(prod)

        # Step 5: Normalize trick
        alpha = prod.prob.sum()
        prod.prob = prod.prob/alpha
        prod.select(query_var)
        result = prod.prob.reshape(1)[0]

        return result

    def __extract_query(self, line) -> (dict, dict):
        parts = line.split(';')

        # extract evidence variables
        evidence_variables = {}
        for item in parts[1].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            val2idx = self.dom2idx[lst[0]][lst[1]]
            evidence_variables[lst[0]] = val2idx

        # extract query variables
        query_variables = {}
        for item in parts[0].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            val2idx = self.dom2idx[lst[0]][lst[1]]
            query_variables[lst[0]] = val2idx

        return query_variables, evidence_variables


def extract_model(line) -> (str, [str], [str], tuple, np.ndarray):
    parts = line.split(';')

    # Name of current node
    node: str = parts[0]

    # Name list of parrent nodes
    if parts[1] == '':
        parents: [str] = []
    else:
        parents: [str] = parts[1].split(',')

    # Domain list of current node
    domain: [str] = parts[2].split(',')

    # Size list of variable list
    shape: (int,) = eval(parts[3])

    # Conditional distribution table (row major)
    probabilities: np.ndarray = np.array(eval(parts[4]))
    probabilities = probabilities.reshape(shape)

    return node, parents, domain, shape, probabilities


if __name__ == "__main__":
    ABCD = np.random.randn(2, 3, 4, 5)
    EFCD = np.random.randn(6, 7, 4, 5)
    X = Factor(ABCD, ('A', 'B', 'C', 'D'))
    Y = Factor(EFCD, ('E', 'F', 'C', 'D'))
    # Test product factor
    XY = X.product(Y)
    print(XY)
    # Test select factor
    X.select({'B': 1})
    print(X)
    # Test reduce factor
    Y.reduce('F')
    print(Y)
