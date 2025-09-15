from py_irt. import irt

# Responses: rows = students, cols = items (1 = correct, 0 = incorrect)
responses = [
    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 1, 1, 0]
]

# Fit IRT model
theta, a, b = irt(responses, model="2PL")

print("Student abilities (θ):", theta)
print("Item discrimination (a):", a)
print("Item difficulty (b):", b)
