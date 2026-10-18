"""Typed decisions with the TypeSafe SDK, served by a local SIE.

This is ordinary TypeSafe SDK code. Nothing in it names SIE: the SDK reads
TYPESAFE_BASE_URL, TYPESAFE_DEFAULT_MODEL and TYPESAFE_API_KEY from the
environment, and those point it at SIE (see README.md).
"""

from __future__ import annotations

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

TICKET = "I was charged twice for my March invoice. Please refund one of the charges today."


def main() -> None:
    with TypeSafeClient() as client:
        response = client.system_one(
            state=TICKET,
            questions={
                "spam": Noul(instructions="Is this message unsolicited spam?"),
                "intent": Choice(
                    instructions="What does the customer want?",
                    criteria={
                        "billing": "payments, invoices or refunds",
                        "technical": "something is broken",
                        "other": None,
                    },
                ),
                "urgency": Score(
                    instructions="How urgent is this request?",
                    criteria=["can wait", "this week", "today"],
                ),
            },
        )

    print(f"model: {response.model}")
    print(f"spam: P(yes) = {response.nouls['spam'].noul:.2f}")
    intent = response.choices["intent"]
    print(f"intent: {intent.choice} (confidence {intent.confidence:.2f})")
    urgency = response.scores["urgency"]
    print(f"urgency: {urgency.score:.2f} of 0-2 (confidence {urgency.confidence:.2f})")
    print(f"input tokens: {response.usage.input_tokens}")


if __name__ == "__main__":
    main()
