import pandas as pd
import numpy as np
import random
import os
import pickle
from collections import Counter
from bisect import bisect_left


class PriorityQueue(object):
    """Keep a certain number of elements in order.
    Attributes:
        items: a list in the form of [(key, value)].
            the elements are ordered by value in an ascending order.
        size: the maximum number of the elements.
    """

    def __init__(self, size, items=[]):
        """Init the class"""
        self.items = items.copy()
        self.size = size

    def __len__(self):
        """"Return the number of elements in the list"""
        return len(self.items)

    def _insert(self, key, val):
        """Insert a new item in the list or change the value of an existing item."""
        index = bisect_left([x[1] for x in self.items], val)
        self.items.insert(index, (key, val))
        if len(self.items) > self.size:
            self.items.pop(0)

    def insert(self, key, val):
        if (key, val) not in self.items:
            if key in [x[0] for x in self.items]:
                self.items = [item for item in self.items if item[0] != key]
            index = bisect_left([x[1] for x in self.items], val)
            self.items.insert(index, (key, val))
            if len(self.items) > self.size:
                self.items.pop(0)

    def pop(self):
        """Delete and return the element with largest value"""
        return self.items.pop()

    def size_down(self):
        """Shrink the size of the list"""
        self.size -= 1

    def copy(self):
        """Return a copy of the list"""
        return PriorityQueue(self.size, self.items)


class ORIC(object):
    """Get main effects and interactive features.

    Attributes:
        n_freq: the number of the frequent itemsets, an integer.
        n_conf: the number of the confident rules, an integer.
        max_depth: the maximum depth of a chain, an integer.
        max_size: the maximum size of a frequent itemset, an integer.
        n_chain: the number of chains, an integer.
        binary: bool, standing for whether the features are binary.
            If true, only items like "X=1" are concerned,
        positive_class: bool. If true, only rules for positive class will be considered.
        online: bool, indicates whether the style is online
        decay: float, the decay rate for the history records, only used for online style.
        num_cls: a dict, records the number of instances for different classes.
        frequent_set: the frequent itemsets for different classes,
            a dict in the form of {label: orderlist},
            where the elements in the orderlist is (item, freq),
            item is in the form of ((index, value),...),
            (index, value) stands for the item "X[index]==value",
            freq is the corresponding frequency, a float.
        confident_rule: the confident rules for different classes,
            an orderlist with elements as ((item, c), conf),
            a dict in the form of {label: orderlist},
            where the elements in the orderlist is (item, conf),
            item is in the form of ((index, value),...),
            (index, value) stands for the item "X[index]==value",
            conf is the corresponding confidence, a float.
        rules: a list of rules, with elements in the form of (antecedents, consequence),
            where antecedents is a tuple with elements in the form of (feature, value),
            (feature, value) stands for the item "X[feature]==value",
            consequence is in the form of (target, c).
        new_features: a list of feature names of the generated features.
    """

    def __init__(self, n_freq, n_conf, max_depth=1000, max_size=4, n_chain=300,
                 binary=False, positive_class=False, online=False, decay=1, random_state=2020):
        """Init class with features that need to be encoded"""
        self.max_depth = max_depth
        self.max_size = max_size
        self.n_chain = n_chain
        self.n_freq = n_freq
        self.n_conf = n_conf
        self.binary = binary
        self.positive_class = positive_class
        self.online = online
        self.decay = decay
        self.t = 0
        self.num_cls = {}
        self.freqs = None
        self.frequent_set = None
        self.confident_rule = None
        self.rules = None
        self.new_features = None
        self.random_state = random_state
        random.seed(random_state)

    def generate_chains(self, X, y, miss_val):
        """Generate chains for different classes
        Args:
            X: the design matrix, a 2-D numpy.array.
            y: the labels, an 1-D numpy.array.

        Returns:
            a dict in the form of {c: [chain,...]}
            where c is a label, an integer,
            chain a tuple in the form of (values, counts),
            values is a line in the design matrix, an np.array,
            counts is a list of integers, recording how many times the value occurs.
        """

        def generate_chain(X, miss_val):
            """Generate a chain
            Args:
                X: the design matrix, a 2-D numpy.array.
            Returns:
                a tuple in the form of (values, counts),
            """
            value = X[random.randint(0, len(X) - 1)]
            if self.binary:
                count = value.copy()
            else:
                count = [1] * len(value)
            count = np.where(np.isin(value, miss_val), 0, count)
            size = sum(count)
            i = 1
            while i < self.max_depth and size > self.max_size:
                # sample new an instance unless the maximum depth is reached
                # or the number of remained items is sufficiently small
                new_value = X[random.randint(0, len(X) - 1)]
                for p, val in enumerate(new_value):
                    if count[p] == i:
                        if value[p] == val:
                            count[p] += 1
                        else:
                            size -= 1
                i += 1
            return value, count

        chains = {c: [] for c in np.unique(y)}
        for c in chains:
            x_c = X[y == c]
            for _ in range(self.n_chain):
                chains[c].append(generate_chain(x_c, miss_val))
        return chains

    def update_frequency(self, item, chains):
        """Update the frequency of an item"""

        def max_depth(item, chain):
            """Return the maximum depth of an item in a chain"""
            return min([chain[1][i] if chain[0][i] == val else 0 for i, val in item])

        for c in chains:
            K, M, T = self.freqs[c][item] if item in self.freqs[c] else (0, 0, 0)
            while T < self.t - 1:
                K *= self.decay
                M = M * self.decay + self.n_chain
                T += 1
            if T < self.t:
                K *= self.decay
                M *= self.decay
                for chain in chains[c]:
                    k = max_depth(item, chain)
                    K += k
                    if k < max(chain[1]) or max(chain[1]) == 0:
                        M += 1
                self.freqs[c][item] = (K, M, self.t)

    def get_frequency(self, item, c):
        "Return the frequency of an item for class c."
        if c not in self.freqs or item not in self.freqs[c]:
            return 0
        K, M, _ = self.freqs[c][item] if item in self.freqs[c] else (0, 0, 0)
        return K / (K + M) if (K + M) > 0 else 0

    def select_frequent_subset(self, temp_set, c, chains):
        """Select the most frequent subsets of temp_set.
        Args:
            temp_set: a tuple of items.
            c: the label.
            chains: dict {c: ((value, count),...)}
        """
        temp_set = set(temp_set)
        for x in temp_set:
            item = (x,)
            self.update_frequency(item, chains)
            freq = self.get_frequency(item, c)
            self.frequent_set[c].insert(item, freq)

        max_size = sum(len(item[0]) == 1 and item[0][0] in temp_set for item in self.frequent_set[c].items)
        max_size = min(max_size, self.max_size)
        for i in range(2, max_size + 1):
            token_a = self.frequent_set[c].copy()
            while len(token_a) > 1:
                a = set(token_a.pop()[0])
                token_a.size_down()
                if len(a) == 1 and a.issubset(temp_set):
                    token_b = token_a.copy()
                    while len(token_b) >= 1:
                        b = set(token_b.pop()[0])
                        token_b.size_down()
                        if len(b) == i - 1 and b.issubset(temp_set) and len(a & b) == 0:
                            item = tuple(sorted(list(a | b)))
                            self.update_frequency(item, chains)
                            freq = self.get_frequency(item, c)
                            self.frequent_set[c].insert(item, freq)
                            token_a.insert(item, freq)
                            token_b.insert(item, freq)

    def select_frequent_set(self, chains):
        """Extract frequent itemsets from chains and calculate thier frequency.
           The results are stored in self.frequent_set.
        Args:
            chains: a list of (value, count).
        """
        for c in chains:
            for item, count in chains[c]:
                depth = max(count)
                max_freq_item = tuple((i, item[i]) for i in range(len(count)) if count[i] == depth)
                # print(max_freq_item)
                if depth:
                    self.select_frequent_subset(max_freq_item, c, chains)

    def calculate_confidence(self, item, freq, c, prior, margin):
        """Calculate the confidence of an item.
        Args:
            item: a tuple of (index, value).
            freq: the frequency of the item, a float.
            c: the label.
            chains: a list of (value, count).
            prior: the probability of each class,
                a dict in the form of {c: freq}.
            margin: the marginal probability of the item.
                a dict in the form of {item: prob}.
        Returns:
            the confidence of the item.
        """

        # calculate the marginal probability
        if item not in margin:
            margin[item] = sum([prior[_] * self.get_frequency(item, _)
                                for _ in prior])
        conf = prior[c] * freq / margin[item]
        return conf

    def select_confident_rule(self, prior):
        """Extract rules from frequent itemsets and calculate thier confidence.
           The results are stored in self.confident_rule.
        Args:
            chains: a list of (value, count).
            prior: the probability of each class,
                a dict in the form of {c: freq}.
        """

        def better_interact(item, conf, confident_rules):
            """Return true only if each subset of the input item
               is not in confident_rules or less confident than item.
            """
            for item_old, conf_old in confident_rules:
                if set(item_old).issubset(set(item)) and conf_old > 1 * conf:
                    return False
            return True

        margin = {}
        for c in prior:
            for item, freq in self.frequent_set[c].items[::-1]:
                conf = self.calculate_confidence(item, freq, c, prior, margin)
                if better_interact(item, conf, self.confident_rule[c].items):
                    self.confident_rule[c].insert(item, conf)

    def generate_rule(self, features, prior):
        """Generate rules."""

        def relative_conf(conf, prior):
            return conf / (1.0001 - conf) * (1 - prior) / (prior + 0.0001)

        for c in self.confident_rule:
            if self.positive_class and c == 0:
                continue
            for item, conf in self.confident_rule[c].items:
                antecedents = tuple(map(lambda x: (features[x[0]], x[1]), item))
                consequence = c
                rconf = relative_conf(conf, prior[c])
                self.rules.insert((antecedents, consequence), rconf)

    def generate_feature_name(self):
        """Generate feature names for the selected itemsets."""
        for (antecedents, _), _ in self.rules.items:
            feature_name = '__'.join(["{}-{}".format(f, val) for f, val in antecedents])
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)

    def new_period(self):
        """Start a new period."""
        self.t += 1

    def fit(self, X, y, miss_val=[]):
        """Get the useful features.

        Args:
            X: input data set, a DataFrame.
            y: labels, an 1d np.array.
            miss_val: a list, the labels of missing values.
        """
        # process the input
        features = list(X.columns)
        X = X.values
        classes = np.unique(y)

        # init
        self.new_features = []
        self.t += 1
        if self.t == 1:
            self.freqs = {c: {} for c in classes}
            self.frequent_set = {c: PriorityQueue(self.n_freq) for c in classes}
        self.confident_rule = {c: PriorityQueue(self.n_conf) for c in classes}
        self.rules = PriorityQueue(self.n_conf)

        # calculate prior possibilities
        if self.online:
            self.num_cls = Counter(y)
        else:
            for c, count in Counter(y).items():
                self.num_cls[c] = self.num_cls.get(c, 0) + count
        num_total = sum([self.num_cls[c] for c in self.num_cls])
        prior = {c: self.num_cls[c] / num_total for c in self.num_cls}

        # generate chains
        chains = self.generate_chains(X, y, miss_val)

        # select frequent itemsets
        for c in classes:
            frequent_set_old = self.frequent_set[c].items
            for item, _ in frequent_set_old:
                self.update_frequency(item, chains)
                freq = self.get_frequency(item, c)
                self.frequent_set[c].insert(item, freq)
        self.select_frequent_set(chains)

        # select confident rules
        self.select_confident_rule(prior)

        # generate feature names
        self.generate_rule(features, prior)
        self.generate_feature_name()

    def transform(self, X):
        """Return the DataFrame of associated features.

        Args:
            X: the design matrix, a pd.DataFrame.
        Returns:
            a DataFrame contains generated features.
        """
        res = {}
        for (antecedents, _), _ in self.rules.items:
            feature_name = '__'.join(["{}-{}".format(f, val) for f, val in antecedents])
            if feature_name not in res:
                idx = [True] * len(X)
                for f, val in antecedents:
                    idx &= (X[f]==val)
                res[feature_name] = np.where(idx, 1, 0)
        return pd.DataFrame(res, index=X.index, columns=self.new_features, dtype=np.int8)

    def transform_inter(self, X):
        """Return the DataFrame of interactive features.

        Args:
           X: the design matrix, a pd.DataFrame.
        Returns:
            a DataFrame contains generated interactive effects.
        """
        res = {}
        for (antecedents, _), _ in self.rules.items:
            feature_name = '__'.join(["{}-{}".format(f, val) for f, val in antecedents])
            if feature_name not in res and len(antecedents) > 1:
                idx = [True] * len(X)
                for f, val in antecedents:
                    idx &= (X[f]==val)
                res[feature_name] = np.where(idx, 1, 0)
        return pd.DataFrame(res, index=X.index, dtype=np.int8)

    def save(self, path):
        """Save the ORIC model."""
        with open(path, "wb") as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_oric(path):
    """Load the ORIC model"""
    if os.path.exists(path):
        with open(path, "rb") as handle:
            attr_dict = pickle.load(handle)
        oric = ORIC(0, 0)
        for attr in attr_dict:
            oric.__dict__[attr] = attr_dict[attr]
        return oric
    else:
        print("file {} does not exist".format(path))
        return None
