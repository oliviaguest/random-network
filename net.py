import numpy as np

prototypes  = np.random.randn(2, 100) # two toy categories
members = 10 # how many items per category
patterns = []

for p in prototypes:
    for i, m in enumerate(range(members)):
        # for each item, create a pattern that has noise as a function of the
        # number of items. First item in category has no noise, then 0.05 SD of
        # noise, then 0.1 SD, and so on.
        patterns.append(p + 0.01 * i * np.random.randn((len(p))))

layers = 20 # how many layers we want, i.e., how deep is the network
w = np.random.randn(layers, len(prototypes[0]), len(prototypes[0])) * 0.1 # random weights from one layer to the next, n

for pat in patterns:
    # for each pattern
    for i, l in enumerate(range(layers)):
        if i == 0:
            #if we are at the input layer, then set units to pattern
            n = pat
        # propagate through each layer
        n = np.dot(n, w[i]) # pre-synaptic states in n
        n = np.tanh(n) # post-synaptic states in n
        if i == layers-1:
            # print the layer, the first five features of the pattern applied at
            # input and the first five activations in the last layer 
            print i, pat[0:5], n[0:5]
