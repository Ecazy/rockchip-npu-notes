#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Fail closed until the reviewed prototype satisfies board admission checks.
set -euo pipefail
cat >&2 <<'MESSAGE'
Board campaign is not implemented and device admission has not passed.
The previous scaffold did not enforce --core-mask/--mode or deploy input assets.
See VALIDATION.md for what evidence may be cited and ARCHITECTURE.md for the
runtime boundary; historical reviews are in Git history. This entry point performs no SSH or
Submit. Use REPLAY.md for current commands.
MESSAGE
exit 2
