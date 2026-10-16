
def scaling_policy(predicted_cpu, current_cpu, current_memory):
    current_instances = 2

    if (
        predicted_cpu >= 70
        or current_cpu >= 80
        or current_memory >= 85
    ):
        decision = "SCALE UP"
        required_instances = current_instances + 1

    elif (
        predicted_cpu >= 40
        or current_cpu >= 40
        or current_memory >= 70
    ):
        decision = "MAINTAIN"
        required_instances = current_instances

    else:
        decision = "SCALE DOWN"
        required_instances = max(1, current_instances - 1)

    return decision, required_instances


test_cases = [
    ("High memory", 28, 24, 88, "SCALE UP", 3),
    ("Moderate CPU", 45, 30, 60, "MAINTAIN", 2),
    ("Low usage", 20, 25, 50, "SCALE DOWN", 1),
]

print("=== SCALING POLICY TESTS ===")

for name, predicted, cpu, memory, expected, instances in test_cases:
    decision, required = scaling_policy(predicted, cpu, memory)

    passed = decision == expected and required == instances
    status = "PASS" if passed else "FAIL"

    print(f"{name}: {status} | {decision} | Instances: {required}")

print("\nAll tests passed:", all(
    scaling_policy(p, c, m) == (expected, instances)
    for _, p, c, m, expected, instances in test_cases
))
