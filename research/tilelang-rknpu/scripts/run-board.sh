#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Fail closed until the reviewed prototype satisfies board admission checks.
set -euo pipefail
cat >&2 <<'MESSAGE'
Board campaign is not implemented and device admission has not passed.
The previous scaffold did not enforce --core-mask/--mode or deploy input assets.
Resolve REVIEW-2026-10-20.md first; this entry point performs no SSH or Submit.
MESSAGE
exit 2
