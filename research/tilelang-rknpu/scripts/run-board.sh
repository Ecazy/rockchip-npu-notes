#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Fail closed until the reviewed prototype satisfies board admission checks.
set -euo pipefail
cat >&2 <<'MESSAGE'
Board campaign is not implemented and device admission has not passed.
The previous scaffold did not enforce --core-mask/--mode or deploy input assets.
See archive/development-history.md#review-2026-10-20 for the historical review;
this entry point performs no SSH or Submit. Use REPLAY.md for current commands.
MESSAGE
exit 2
