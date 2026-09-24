"""Generate the Version 3 semantic test-case dataset.

Version 3 removes the positional shortcut in the earlier dataset. Acceptance
criteria vary from two to five items, appear in shuffled order, and may contain
repeated or missing test types. Complete scenario families are assigned to
train, development, or final test splits.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from generate_dataset import SCENARIOS as ORIGINAL_SCENARIOS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "user_story_test_cases_v3.csv"
EXAMPLES_PER_FAMILY = 20
SEED = 20260922


def typed(case_type: str, text: str) -> dict[str, str]:
    return {"type": case_type, "text": text}


def original_pool(scenario: dict, variant: tuple[str, ...]) -> list[dict[str, str]]:
    positive, negative, boundary = scenario["build"](variant)
    subject = scenario["goal"]
    return [
        typed("positive", positive),
        typed("negative", negative),
        typed("boundary", boundary),
        typed(
            "positive",
            f"Given the user has successfully completed the request to {subject}, "
            "when the result is opened again, then the confirmed outcome remains visible "
            "without creating another record.",
        ),
        typed(
            "negative",
            f"Given the same request to {subject} was already processed, when it is "
            "submitted again, then no duplicate result is created and the existing "
            "outcome is shown.",
        ),
    ]


NEW_SCENARIOS = [
    {
        "family": "refund_request",
        "domain": "ecommerce",
        "split": "development",
        "role": "customer",
        "goal": "request a refund for an eligible purchase",
        "benefit": "I can recover funds when an order qualifies",
        "variants": [
            ("30 days", "$250"), ("14 days", "$100"), ("60 days", "$500"),
            ("7 days", "$75"), ("45 days", "$1,000"),
        ],
        "build": lambda v: [
            typed("positive", f"Given a delivered order is within {v[0]}, when a refund is requested, then a refund case is created and its reference is displayed."),
            typed("negative", f"Given a delivered order is older than {v[0]}, when a refund is requested, then submission is blocked and the eligibility rule is explained."),
            typed("boundary", f"Given a delivered order is exactly {v[0]} old, when a refund is requested, then the configured eligibility boundary is applied consistently."),
            typed("positive", f"Given an approved refund of {v[1]}, when processing completes, then the payment status and refund amount are updated once."),
            typed("negative", "Given a refund already exists for the order, when another request is submitted, then no duplicate refund case is created."),
        ],
    },
    {
        "family": "document_approval",
        "domain": "workflow",
        "split": "development",
        "role": "reviewer",
        "goal": "approve a document through the review workflow",
        "benefit": "the organization has an auditable decision",
        "variants": [
            ("2", "legal"), ("3", "finance"), ("1", "security"),
            ("4", "compliance"), ("2", "operations"),
        ],
        "build": lambda v: [
            typed("positive", f"Given all required {v[1]} reviewers have approved, when final approval is submitted, then the document becomes approved and the audit trail is updated."),
            typed("negative", "Given a required reviewer rejected the document, when final approval is attempted, then approval is blocked and the rejection remains visible."),
            typed("boundary", f"Given exactly {v[0]} approvals are required, when the {v[0]}th approval is recorded, then the document advances exactly once."),
            typed("positive", "Given a document is awaiting review, when an authorized reviewer adds a decision, then the decision and timestamp are saved."),
            typed("negative", "Given a user is not an assigned reviewer, when an approval is submitted, then it is rejected and the audit trail is unchanged."),
        ],
    },
    {
        "family": "delivery_reschedule",
        "domain": "logistics",
        "split": "development",
        "role": "recipient",
        "goal": "reschedule an upcoming delivery",
        "benefit": "the parcel arrives when someone is available",
        "variants": [
            ("2 hours", "3"), ("4 hours", "5"), ("24 hours", "2"),
            ("1 hour", "4"), ("12 hours", "7"),
        ],
        "build": lambda v: [
            typed("positive", f"Given an eligible delivery has more than {v[0]} remaining, when a new date is selected, then the delivery date and confirmation are updated."),
            typed("negative", "Given the parcel is already out for delivery, when rescheduling is attempted, then the request is rejected and the current schedule remains."),
            typed("boundary", f"Given the delivery cutoff is exactly {v[0]} away, when rescheduling is requested, then the cutoff policy is applied and explained."),
            typed("positive", f"Given {v[1]} valid dates are available, when the recipient opens rescheduling, then all {v[1]} dates are shown once."),
            typed("negative", "Given an unavailable date is submitted, when the change is confirmed, then no schedule update occurs and alternatives are shown."),
        ],
    },
    {
        "family": "two_factor_enrollment",
        "domain": "authentication",
        "split": "test",
        "role": "account owner",
        "goal": "enroll a second authentication factor",
        "benefit": "my account has stronger sign-in protection",
        "variants": [
            ("6 digits", "30 seconds"), ("8 digits", "60 seconds"),
            ("6 digits", "90 seconds"), ("7 digits", "45 seconds"),
            ("8 digits", "2 minutes"),
        ],
        "build": lambda v: [
            typed("positive", f"Given a verified account, when a valid {v[0]} enrollment code is submitted, then the factor is activated and recovery codes are displayed once."),
            typed("negative", "Given an incorrect enrollment code, when verification is attempted, then enrollment remains incomplete and an error is shown."),
            typed("boundary", f"Given an enrollment code expires in exactly {v[1]}, when it is submitted at the cutoff, then the configured expiry rule is applied consistently."),
            typed("positive", "Given a factor is active, when the next sign-in succeeds with it, then the authentication event is recorded."),
            typed("negative", "Given the factor is already enrolled, when the same enrollment is repeated, then no duplicate factor or recovery codes are created."),
        ],
    },
    {
        "family": "api_rate_limit",
        "domain": "developer_platform",
        "split": "test",
        "role": "API consumer",
        "goal": "send requests within the service rate limit",
        "benefit": "my integration behaves predictably under load",
        "variants": [
            ("100", "minute"), ("1,000", "hour"), ("20", "second"),
            ("500", "minute"), ("10,000", "day"),
        ],
        "build": lambda v: [
            typed("positive", f"Given the client is below {v[0]} requests per {v[1]}, when another valid request is sent, then it is processed and limit headers are returned."),
            typed("negative", f"Given the client exceeded {v[0]} requests per {v[1]}, when another request is sent, then it is rejected without executing the operation."),
            typed("boundary", f"Given the client has sent exactly {v[0]} requests in the current {v[1]}, when one more is sent, then the documented boundary response and retry time are returned."),
            typed("positive", "Given the rate-limit window has reset, when a valid request is sent, then processing resumes and the remaining count is refreshed."),
            typed("negative", "Given a rejected request is retried before the retry time, when it arrives, then it remains rejected and no duplicate operation occurs."),
        ],
    },
    {
        "family": "inventory_reservation",
        "domain": "warehouse",
        "split": "test",
        "role": "fulfillment coordinator",
        "goal": "reserve inventory for an order",
        "benefit": "available stock is protected until fulfillment",
        "variants": [
            ("10 units", "15 minutes"), ("50 units", "30 minutes"),
            ("1 unit", "5 minutes"), ("100 units", "60 minutes"),
            ("25 units", "20 minutes"),
        ],
        "build": lambda v: [
            typed("positive", f"Given at least {v[0]} are available, when the order is confirmed, then {v[0]} are reserved and available stock decreases once."),
            typed("negative", f"Given fewer than {v[0]} are available, when reservation is attempted, then it is rejected and stock counts remain unchanged."),
            typed("boundary", f"Given reserved stock expires in exactly {v[1]}, when the expiry job runs, then the configured release boundary is applied once."),
            typed("positive", "Given a reservation is active, when fulfillment consumes it, then reserved stock decreases and the order records the allocation."),
            typed("negative", "Given the order already has an active reservation, when the same reservation request repeats, then no duplicate stock allocation is created."),
        ],
    },
]


def make_output(criteria: list[dict[str, str]]) -> str:
    cases = []
    for index, criterion in enumerate(criteria, start=1):
        text = criterion["text"]
        given, when_then = text.split(", when ", 1)
        action, expected = when_then.split(", then ", 1)
        precondition = given.removeprefix("Given ")
        cases.append(
            {
                "id": f"TC-{index:03d}",
                "title": f"Verify AC{index}: {action.rstrip('.')}",
                "type": criterion["type"],
                "preconditions": [precondition],
                "steps": [
                    f"Establish this precondition: {precondition}.",
                    f"{action[0].upper() + action[1:].rstrip('.')}.",
                    "Observe the system response and persisted state.",
                ],
                "expected_result": expected[0].upper() + expected[1:].rstrip(".") + ".",
                "covers": [f"AC{index}"],
            }
        )
    return json.dumps(cases, ensure_ascii=False, separators=(",", ":"))


def select_criteria(pool: list[dict[str, str]], rng: random.Random) -> list[dict[str, str]]:
    count = rng.randint(2, 5)
    selected = rng.sample(pool, count)
    rng.shuffle(selected)
    return selected


def build_rows() -> list[dict[str, str]]:
    rng = random.Random(SEED)
    rows: list[dict[str, str]] = []
    used_inputs: set[tuple[str, str]] = set()

    scenarios = [
        {**scenario, "split": "train", "pool_builder": original_pool}
        for scenario in ORIGINAL_SCENARIOS
    ] + NEW_SCENARIOS

    for scenario in scenarios:
        for example_index in range(1, EXAMPLES_PER_FAMILY + 1):
            variant = scenario["variants"][(example_index - 1) % len(scenario["variants"])]
            if "pool_builder" in scenario:
                pool = scenario["pool_builder"](scenario, variant)
            else:
                pool = scenario["build"](variant)
            for _ in range(100):
                criteria = select_criteria(pool, rng)
                acceptance_criteria = "\n".join(
                    f"AC{i}: {criterion['text']}"
                    for i, criterion in enumerate(criteria, start=1)
                )
                input_key = (scenario["family"], acceptance_criteria)
                if input_key not in used_inputs:
                    used_inputs.add(input_key)
                    break
            else:
                raise RuntimeError(f"Could not create a unique input for {scenario['family']}")
            rows.append(
                {
                    "example_id": f"{scenario['family']}-{example_index:02d}",
                    "scenario_family": scenario["family"],
                    "domain": scenario["domain"],
                    "split": scenario["split"],
                    "user_story": (
                        f"As a {scenario['role']}, I want to {scenario['goal']}, "
                        f"so that {scenario['benefit']}."
                    ),
                    "acceptance_criteria": acceptance_criteria,
                    "expected_test_cases": make_output(criteria),
                }
            )
    return rows


def validate(rows: list[dict[str, str]]) -> None:
    assert len(rows) == 440
    ids = [row["example_id"] for row in rows]
    assert len(ids) == len(set(ids))

    families_by_split: dict[str, set[str]] = {"train": set(), "development": set(), "test": set()}
    type_positions: dict[tuple[int, str], int] = {}
    counts: set[int] = set()

    for row in rows:
        families_by_split[row["split"]].add(row["scenario_family"])
        cases = json.loads(row["expected_test_cases"])
        criteria_count = len(row["acceptance_criteria"].splitlines())
        assert len(cases) == criteria_count
        counts.add(criteria_count)
        for position, case in enumerate(cases, start=1):
            assert case["type"] in {"positive", "negative", "boundary"}
            type_positions[(position, case["type"])] = type_positions.get((position, case["type"]), 0) + 1
            assert case["covers"] == [f"AC{position}"]
            assert isinstance(case["expected_result"], str)

    assert counts == {2, 3, 4, 5}
    assert families_by_split["train"].isdisjoint(families_by_split["development"])
    assert families_by_split["train"].isdisjoint(families_by_split["test"])
    assert families_by_split["development"].isdisjoint(families_by_split["test"])
    for position in (1, 2):
        observed = {case_type for (item_position, case_type) in type_positions if item_position == position}
        assert observed == {"positive", "negative", "boundary"}


def main() -> None:
    rows = build_rows()
    validate(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} examples to {OUTPUT}")
    for split in ("train", "development", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        families = sorted({row["scenario_family"] for row in split_rows})
        print(f"{split}: {len(split_rows)} examples across {len(families)} families")
        print("  ", families)


if __name__ == "__main__":
    main()
