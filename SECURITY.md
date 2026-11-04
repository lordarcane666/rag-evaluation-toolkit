# Security policy

The initial 0.1.x release receives fixes on the main branch. This is an alpha
project; it has not undergone an independent security audit.

If GitHub private vulnerability reporting is available for the repository,
use its Security tab to report issues privately. Otherwise request a private
contact channel with a minimal issue; do not disclose exploit details, private
data, or credentials publicly. No nonexistent security email is implied.

Include the affected version, a synthetic reproduction, expected behavior,
and impact. Scope includes unintended network transmission, credential leaks,
unsafe report output, parser faults, and accidental execution of dataset content.
Do not use other people's datasets or systems to demonstrate a vulnerability.

LLM prompt injection remains a known limitation: schema and quote validation
cannot prove semantic correctness. See [privacy](docs/privacy.md). Third-party
judges/plugins are trusted code and are responsible for their own data handling.
