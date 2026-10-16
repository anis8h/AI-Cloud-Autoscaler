
COST_PER_INSTANCE = 5.0

test_cases = [
    ("SCALE UP", 2, 3, 10.0, 15.0, -5.0, -50.0),
    ("MAINTAIN", 2, 2, 10.0, 10.0, 0.0, 0.0),
    ("SCALE DOWN", 2, 1, 10.0, 5.0, 5.0, 50.0),
]

print("=== COST CALCULATION TESTS ===")

for name, before, after, exp_before, exp_after, exp_saving, exp_pct in test_cases:
    cost_before = before * COST_PER_INSTANCE
    cost_after = after * COST_PER_INSTANCE
    savings = cost_before - cost_after
    savings_percentage = (savings / cost_before) * 100 if cost_before else 0

    passed = (
        cost_before == exp_before
        and cost_after == exp_after
        and savings == exp_saving
        and savings_percentage == exp_pct
    )

    print(
        f"{name}: {'PASS' if passed else 'FAIL'} | "
        f"Before: ₹{cost_before:.2f}/hr | "
        f"After: ₹{cost_after:.2f}/hr | "
        f"Saving: ₹{savings:.2f}/hr ({savings_percentage:.2f}%)"
    )
