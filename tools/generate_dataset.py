"""Generate the synthetic dataset for the Week 5 test-case fine-tuning project.

The file is deterministic so the experiment can be reproduced exactly.
Each scenario family produces five examples. Entire families are assigned to
either train or validation to avoid putting near-duplicates in both splits.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "user_story_test_cases.csv"


SCENARIOS = [
    {
        "family": "login_lockout",
        "domain": "authentication",
        "role": "registered user",
        "goal": "sign in securely",
        "benefit": "I can access my account while repeated attacks are blocked",
        "variants": [
            ("email and password", "5", "15 minutes"),
            ("username and password", "3", "10 minutes"),
            ("employee ID and password", "5", "30 minutes"),
            ("mobile number and PIN", "4", "20 minutes"),
            ("member ID and password", "6", "60 minutes"),
        ],
        "build": lambda v: (
            f"Given an active account, when valid {v[0]} are submitted, then the dashboard is displayed.",
            f"Given an active account, when invalid {v[0]} are submitted, then a generic authentication error is displayed.",
            f"Given {v[1]} consecutive failed attempts, when another sign-in is attempted, then the account is locked for {v[2]}.",
        ),
    },
    {
        "family": "password_reset",
        "domain": "authentication",
        "role": "user who forgot my password",
        "goal": "reset it with a one-time link",
        "benefit": "I can regain access without support assistance",
        "variants": [
            ("15 minutes", "8 characters"), ("30 minutes", "10 characters"),
            ("20 minutes", "12 characters"), ("10 minutes", "14 characters"),
            ("60 minutes", "16 characters"),
        ],
        "build": lambda v: (
            f"Given a registered email, when a reset is requested, then a single-use link valid for {v[0]} is sent.",
            "Given an unknown email, when a reset is requested, then the same neutral confirmation is shown and no email is sent.",
            f"Given a valid reset link, when a new password shorter than {v[1]} is submitted, then the password is rejected.",
        ),
    },
    {
        "family": "shopping_cart",
        "domain": "ecommerce",
        "role": "shopper",
        "goal": "manage products in my cart",
        "benefit": "I can review an accurate order before checkout",
        "variants": [
            ("10", "99"), ("5", "25"), ("20", "50"), ("3", "10"), ("12", "30"),
        ],
        "build": lambda v: (
            "Given an in-stock product, when it is added to the cart, then the item, quantity, and current price appear in the cart.",
            "Given a product with zero stock, when it is added, then the cart is unchanged and an out-of-stock message appears.",
            f"Given a product in the cart, when quantity is changed to the maximum of {v[0]}, then the cart accepts it; quantity {v[1]} is rejected.",
        ),
    },
    {
        "family": "checkout_coupon",
        "domain": "ecommerce",
        "role": "shopper",
        "goal": "apply a promotion at checkout",
        "benefit": "I can receive the discount for which my order qualifies",
        "variants": [
            ("SAVE10", "$50", "10%"), ("SHIPFREE", "$75", "free shipping"),
            ("WELCOME15", "$40", "15%"), ("SPRING20", "$100", "20%"),
            ("MEMBER5", "$25", "5%"),
        ],
        "build": lambda v: (
            f"Given an eligible subtotal of at least {v[1]}, when code {v[0]} is applied, then {v[2]} is reflected in the order total.",
            f"Given a subtotal below {v[1]}, when code {v[0]} is applied, then no discount is added and an eligibility message appears.",
            f"Given code {v[0]} has expired, when it is applied, then the order total is unchanged and an expired-code message appears.",
        ),
    },
    {
        "family": "bank_transfer",
        "domain": "banking",
        "role": "banking customer",
        "goal": "transfer money between my accounts",
        "benefit": "I can move funds safely and see accurate balances",
        "variants": [
            ("$1", "$5,000"), ("$5", "$10,000"), ("$0.01", "$2,500"),
            ("$10", "$20,000"), ("$25", "$1,000"),
        ],
        "build": lambda v: (
            f"Given sufficient funds, when a transfer from {v[0]} through {v[1]} is confirmed, then both balances update and one transaction record is created.",
            "Given insufficient funds, when a transfer is confirmed, then it is rejected and neither balance changes.",
            f"Given sufficient funds, when exactly the maximum amount {v[1]} is transferred, then the transfer succeeds; any larger amount is rejected.",
        ),
    },
    {
        "family": "invoice_export",
        "domain": "finance",
        "role": "accounts analyst",
        "goal": "export filtered invoices",
        "benefit": "I can reconcile billing data offline",
        "variants": [
            ("CSV", "10,000"), ("XLSX", "5,000"), ("CSV", "25,000"),
            ("XLSX", "2,000"), ("CSV", "50,000"),
        ],
        "build": lambda v: (
            f"Given matching invoices, when export format {v[0]} is selected, then the downloaded file contains only filtered rows and visible columns.",
            "Given no invoices match the filters, when export is requested, then no empty file is created and a no-results message appears.",
            f"Given exactly {v[1]} matching invoices, when export is requested, then one complete file is produced without duplicate rows.",
        ),
    },
    {
        "family": "appointment_booking",
        "domain": "healthcare",
        "role": "patient",
        "goal": "book an available appointment",
        "benefit": "I can reserve care without calling the clinic",
        "variants": [
            ("30 minutes", "24 hours"), ("45 minutes", "12 hours"),
            ("20 minutes", "48 hours"), ("60 minutes", "72 hours"),
            ("15 minutes", "6 hours"),
        ],
        "build": lambda v: (
            f"Given an available slot, when the patient confirms it, then a {v[0]} appointment is created and a confirmation is sent.",
            "Given a slot was booked by another patient, when confirmation is attempted, then no appointment is created and refreshed alternatives appear.",
            f"Given an appointment starts in exactly {v[1]}, when cancellation is requested, then cancellation follows the clinic cutoff rule and its outcome is shown.",
        ),
    },
    {
        "family": "prescription_refill",
        "domain": "healthcare",
        "role": "patient",
        "goal": "request a prescription refill",
        "benefit": "my clinician can review the request before medicine runs out",
        "variants": [
            ("7 days", "2"), ("10 days", "1"), ("5 days", "3"),
            ("14 days", "2"), ("3 days", "1"),
        ],
        "build": lambda v: (
            f"Given an active prescription expiring within {v[0]}, when a refill is requested, then the request is sent to the prescribing clinician.",
            "Given a discontinued prescription, when a refill is requested, then submission is blocked and the patient is told to contact the clinic.",
            f"Given {v[1]} refill requests are already pending for the prescription, when another is submitted, then no duplicate request is created.",
        ),
    },
    {
        "family": "task_assignment",
        "domain": "project_management",
        "role": "project manager",
        "goal": "assign tasks to team members",
        "benefit": "ownership and workload are visible",
        "variants": [
            ("20", "email"), ("50", "in-app"), ("10", "email and in-app"),
            ("100", "in-app"), ("25", "email"),
        ],
        "build": lambda v: (
            f"Given an active project member, when a task is assigned, then the assignee is saved and receives an {v[1]} notification.",
            "Given a user outside the project, when assignment is attempted, then it is blocked and task ownership is unchanged.",
            f"Given a task already has {v[0]} assignees, when one more is added, then the configured assignee limit is enforced without losing existing assignees.",
        ),
    },
    {
        "family": "file_upload",
        "domain": "collaboration",
        "role": "workspace member",
        "goal": "upload a file to a shared project",
        "benefit": "the team can collaborate on the latest artifact",
        "variants": [
            ("PDF", "25 MB"), ("PNG", "10 MB"), ("DOCX", "50 MB"),
            ("CSV", "5 MB"), ("ZIP", "100 MB"),
        ],
        "build": lambda v: (
            f"Given edit permission, when a valid {v[0]} file smaller than {v[1]} is uploaded, then it appears once with the uploader and timestamp.",
            "Given view-only permission, when upload is attempted, then it is blocked and no file is stored.",
            f"Given edit permission, when a {v[0]} file exactly {v[1]} is uploaded, then it succeeds; a larger file is rejected.",
        ),
    },
    {
        "family": "notification_preferences",
        "domain": "communications",
        "role": "account holder",
        "goal": "control notification channels",
        "benefit": "I receive useful alerts without unwanted messages",
        "variants": [
            ("email", "security alerts"), ("SMS", "payment alerts"),
            ("push", "task reminders"), ("email", "weekly summaries"),
            ("SMS", "delivery updates"),
        ],
        "build": lambda v: (
            f"Given {v[0]} is enabled for {v[1]}, when the relevant event occurs, then one {v[0]} notification is sent.",
            f"Given {v[0]} is disabled for {v[1]}, when the event occurs, then no {v[0]} notification is sent.",
            f"Given the preference is toggled twice quickly, when it is saved, then the final {v[0]} setting persists without duplicate notifications.",
        ),
    },
    {
        "family": "search_filters",
        "domain": "content_platform",
        "role": "content user",
        "goal": "filter search results",
        "benefit": "I can find relevant items quickly",
        "variants": [
            ("date", "100"), ("category", "250"), ("owner", "50"),
            ("status", "500"), ("tag", "1,000"),
        ],
        "build": lambda v: (
            f"Given indexed content, when a valid {v[0]} filter is applied, then every displayed result satisfies that filter.",
            "Given mutually exclusive filters, when search runs, then an empty state appears without stale results.",
            f"Given exactly {v[1]} matching items, when the filter is applied, then pagination exposes every item once while preserving the filter.",
        ),
    },
    {
        "family": "subscription_upgrade",
        "domain": "saas_billing",
        "role": "workspace owner",
        "goal": "upgrade the subscription plan",
        "benefit": "the team can use additional features immediately",
        "variants": [
            ("Basic", "Pro", "10"), ("Starter", "Business", "25"),
            ("Free", "Team", "5"), ("Pro", "Enterprise", "100"),
            ("Monthly", "Annual", "50"),
        ],
        "build": lambda v: (
            f"Given a valid payment method, when {v[0]} is upgraded to {v[1]} for {v[2]} seats, then prorated cost is shown and new features activate after confirmation.",
            "Given a declined payment method, when upgrade is confirmed, then the current plan remains active and no entitlement changes.",
            f"Given exactly {v[2]} licensed seats are occupied, when the upgrade completes, then all existing members retain access exactly once.",
        ),
    },
    {
        "family": "report_date_range",
        "domain": "analytics",
        "role": "operations analyst",
        "goal": "run a report for a date range",
        "benefit": "I can measure performance for the selected period",
        "variants": [
            ("31 days", "UTC"), ("90 days", "America/Los_Angeles"),
            ("365 days", "Europe/London"), ("7 days", "Asia/Kolkata"),
            ("30 days", "Australia/Sydney"),
        ],
        "build": lambda v: (
            f"Given available data, when a range of up to {v[0]} is submitted, then the report includes both boundary dates in {v[1]}.",
            "Given the start date is after the end date, when the report is requested, then validation prevents execution and explains the error.",
            f"Given a range exactly {v[0]} long, when the report runs, then it completes without omitting either boundary date.",
        ),
    },
    {
        "family": "seat_booking",
        "domain": "travel",
        "role": "traveler",
        "goal": "select a seat for my trip",
        "benefit": "my preference is reserved before departure",
        "variants": [
            ("airline", "24 hours"), ("rail", "2 hours"), ("coach", "1 hour"),
            ("ferry", "4 hours"), ("shuttle", "30 minutes"),
        ],
        "build": lambda v: (
            f"Given an available {v[0]} seat, when it is selected and confirmed, then it is assigned to the traveler and removed from availability.",
            "Given a seat is already occupied, when selection is attempted, then assignment is blocked and the current seat remains unchanged.",
            f"Given departure is exactly {v[1]} away, when a seat change is requested, then the booking cutoff rule is applied and clearly shown.",
        ),
    },
    {
        "family": "video_progress",
        "domain": "learning_platform",
        "role": "learner",
        "goal": "resume a course video",
        "benefit": "I can continue from where I stopped",
        "variants": [
            ("5 seconds", "95%"), ("10 seconds", "90%"), ("3 seconds", "100%"),
            ("15 seconds", "80%"), ("1 second", "98%"),
        ],
        "build": lambda v: (
            f"Given playback has advanced more than {v[0]}, when the learner leaves, then the latest position is saved and restored on return.",
            "Given the video has never been played, when it is opened, then playback starts at the beginning without a resume prompt.",
            f"Given watched progress is exactly {v[1]}, when playback ends, then completion status follows the configured threshold and is recorded once.",
        ),
    },
]


# These complete scenario families are never included in fine-tuning.
VALIDATION_FAMILIES = {
    "appointment_booking",
    "notification_preferences",
    "report_date_range",
    "video_progress",
}


def make_output(criteria: list[str]) -> str:
    case_types = ["positive", "negative", "boundary"]
    cases = []
    for index, (criterion, case_type) in enumerate(zip(criteria, case_types), start=1):
        given, when_then = criterion.split(", when ", 1)
        action, expected = when_then.split(", then ", 1)
        precondition = given.removeprefix("Given ")
        cases.append(
            {
                "id": f"TC-{index:03d}",
                "title": f"Verify AC{index}: {action.rstrip('.')}",
                "type": case_type,
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


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        split = "validation" if scenario["family"] in VALIDATION_FAMILIES else "train"
        for variant_index, variant in enumerate(scenario["variants"], start=1):
            criteria = list(scenario["build"](variant))
            rows.append(
                {
                    "example_id": f"{scenario['family']}-{variant_index:02d}",
                    "scenario_family": scenario["family"],
                    "domain": scenario["domain"],
                    "split": split,
                    "user_story": (
                        f"As a {scenario['role']}, I want to {scenario['goal']}, "
                        f"so that {scenario['benefit']}."
                    ),
                    "acceptance_criteria": "\n".join(
                        f"AC{i}: {criterion}" for i, criterion in enumerate(criteria, start=1)
                    ),
                    "expected_test_cases": make_output(criteria),
                }
            )

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    train_count = sum(row["split"] == "train" for row in rows)
    val_count = len(rows) - train_count
    print(f"Wrote {len(rows)} examples to {OUTPUT}")
    print(f"Train: {train_count}; validation: {val_count}")


if __name__ == "__main__":
    main()
