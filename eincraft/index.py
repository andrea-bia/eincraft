class IndexMap:
    """Union-find structure to manage contraction indices."""

    def __init__(self, mapping=None):
        self.parent = {}
        self.nodes = set()
        if mapping:
            if isinstance(mapping, IndexMap):
                mapping = dict(mapping.items())
            for k, v in mapping.items():
                self.add(k, v)

    def add(self, a, b):
        """Union indices *a* and *b* into the same equivalence class."""
        self.nodes.add(a)
        self.nodes.add(b)
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb

    def find(self, idx):
        """Return canonical representative for *idx*."""
        self.nodes.add(idx)
        path = []
        while idx in self.parent:
            path.append(idx)
            idx = self.parent[idx]
        for p in path:
            self.parent[p] = idx
        return idx

    def items(self):
        for n in self.nodes:
            yield n, self.find(n)

    def keys(self):
        return list(self.nodes)

    def values(self):
        return [self.find(n) for n in self.nodes]

    def __contains__(self, idx):
        return idx in self.nodes

    def __getitem__(self, idx):
        return self.find(idx)

    def get(self, idx, default=None):
        if idx in self.nodes:
            return self.find(idx)
        return default

    def copy(self):
        return IndexMap({k: v for k, v in self.items()})

    def update(self, mapping):
        if isinstance(mapping, IndexMap):
            mapping = dict(mapping.items())
        for k, v in mapping.items():
            self.add(k, v)

    def __or__(self, other):
        result = self.copy()
        result.update(other)
        return result

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def to_dict(self):
        return {k: v for k, v in self.items()}

    def __repr__(self):
        return f"IndexMap({self.to_dict()})"

__all__ = ["IndexMap"]
