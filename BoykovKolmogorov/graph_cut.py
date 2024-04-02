from collections import deque
from operator import itemgetter

import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network


class BoykovKolmorogov():

    def __init__(self, G, s, t, capacity, resisual=None, cutoff=None, return_intermediate=False):
        self.s = s
        self.t = t
        self.capacity = capacity

        self.St = {s: None}
        self.Tt = {t: None}
        self.A = deque([s, t])
        self.O = deque()
        
        self.flow_value = 0
        self.dist = {s: 0, t: 0}
        self.current_step = 1
        self.steps = {s: 0, t: 0}

        if resisual is None:
            self.residual = build_residual_network(G, capacity)
        else:
            self.residual = resisual

        for p in self.residual:
            for e in self.residual[p].values():
                e['flow'] = 0

        self.inf = self.residual.graph['inf']
        if cutoff is None:
            self.cutoff = self.inf

        self.succ = self.residual.succ
        self.pred = self.residual.pred
        
        
    def grow(self):
        while self.A:
            u = self.A[0]

            if u in self.St:
                current = self.St
                other = self.Tt
                neighbors = self.succ
            else:
                current = self.Tt
                other = self.St
                neighbors = self.pred

            for v, attr in neighbors[u].items():
                if attr['capacity'] > attr['flow']:
                    if v not in current:
                        if v in other:
                            return (u, v) if current is self.St else (v, u)
                        
                        current[v] = u
                        self.dist[v] = self.dist[u] + 1
                        self.steps[v] = self.steps[u]
                        self.A.append(v)

                    elif v in current and self.steps[v] <= self.steps[u] and self.dist[v] > self.dist[u] + 1:
                        current[v] = u
                        self.dist[v] = self.dist[u] + 1
                        self.steps[v] = self.steps[u]

            _ = self.A.popleft()
        return None, None


    def augment(self, u, v):
        attr = self.succ[u][v]
        flow = min(self.inf, attr['capacity'] - attr['flow'])

        # Trace a path from u to s in St
        path = [u]
        w = u
        while w != self.s:
            n = w
            w = self.St[n]
            attr = self.pred[n][w]
            flow = min(flow, attr['capacity'] - attr['flow'])
            path.append(w)
        path.reverse()

        # Trace a path from v to t in Tt
        path.append(v)
        w = v
        while w != self.t:
            n = w
            w = self.Tt[n]
            attr = self.succ[n][w]
            flow = min(flow, attr['capacity'] - attr['flow'])
            path.append(w)

        # Augment flow along the path and check for orphans
        it = iter(path)
        u = next(it)
        new_orphans = []
        for v in it:
            self.succ[u][v]['flow'] += flow
            self.succ[v][u]['flow'] -= flow
            if self.succ[u][v]['flow'] == self.succ[u][v]['capacity']:
                if v in self.St:
                    self.St[v] = None
                    new_orphans.append(v)
                if u in self.Tt:
                    self.Tt[u] = None
                    new_orphans.append(u)
            u = v
        self.O.extend(sorted(new_orphans, key=self.dist.get))
        
        self.flow_value += flow


    def adopt(self):
        while self.O:
            u = self.O.popleft()
            if u in self.St:
                current = self.St
                neighbors = self.pred
            else:
                current = self.Tt
                neighbors = self.succ
            nbrs = ((n, attr, self.dist[n]) for n, attr in neighbors[u].items()
                    if n in current)
            for v, attr, d in sorted(nbrs, key=itemgetter(2)):
                if attr['capacity'] > attr['flow']:
                    if self.valid_root(v, current):
                        current[u] = v
                        self.dist[u] = self.dist[v] + 1
                        self.steps[u] = self.current_step
                        break
            else:
                nbrs = ((n, attr, self.dist[n]) for n, attr in neighbors[u].items()
                        if n in current)
                for v, attr, d in sorted(nbrs, key=itemgetter(2)):
                    if attr['capacity'] > attr['flow']:
                        if v not in self.A:
                            self.A.append(v)
                    if current[v] == u:
                        current[v] = None
                        self.O.appendleft(v)
                if u in self.A:
                    self.A.remove(u)
                del current[u]


    def valid_root(self, n, tree):
        path = []
        v = n
        while v is not None:
            path.append(v)
            if v == self.s or v == self.t:
                base_dist = 0
                break
            elif self.steps[v] == self.current_step:
                base_dist = self.dist[v]
                break
            v = tree[v]
        else:
            return False
        length = len(path)
        for i, u in enumerate(path, 1):
            self.dist[u] = base_dist + length - i
            self.steps[u] = self.current_step
        return True
    

    def max_flow(self):
        while self.flow_value < self.cutoff:
            p, q = self.grow()
            if p is None:
                break
            self.current_step += 1
            self.augment(p, q)
            self.adopt()
        
        self.residual.graph['flow_value'] = self.flow_value
        self.residual.graph['trees'] = (self.St, self.Tt)
        return self.residual