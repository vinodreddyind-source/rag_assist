"""
Lightweight shared-passcode gate -- NOT real authentication, deliberately.
For sharing a demo link with a specific interviewer during a specific
conversation, not for standing up a public service with real users.

Real auth (OAuth, per-user API keys, session management) would be the
honest answer for anything beyond "one link, one demo, one sitting" --
say that plainly if asked, don't oversell this as production auth.
"""

import os
from dotenv import load_dotenv

load_dotenv()

DEMO_PASSCODE = os.environ.get("DEMO_PASSCODE")  # None = gate disabled entirely


def passcode_required() -> bool:
    """Gate is only active if DEMO_PASSCODE is actually set -- local dev
    with no passcode configured just works normally, no surprise lockout."""
    return bool(DEMO_PASSCODE)


def check_passcode(provided: str | None) -> bool:
    if not passcode_required():
        return True
    return provided == DEMO_PASSCODE


if __name__ == "__main__":
    # Case 1: no passcode configured -- gate disabled, everything passes
    DEMO_PASSCODE = None
    assert passcode_required() is False
    assert check_passcode(None) is True
    assert check_passcode("anything") is True
    print("PASS: gate disabled when DEMO_PASSCODE not set -- no surprise lockout in local dev")

    # Case 2: passcode configured -- correct passcode passes, wrong/missing fails
    DEMO_PASSCODE = "letmein123"
    assert passcode_required() is True
    assert check_passcode("letmein123") is True
    assert check_passcode("wrong") is False
    assert check_passcode(None) is False
    print("PASS: gate enforces the passcode correctly when configured")
