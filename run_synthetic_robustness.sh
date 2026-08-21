#!/usr/bin/env bash
set -euo pipefail

# Run this script from the posterior-landscape package root.
PYTHON_BIN="${PYTHON_BIN:-python}"
COMMON_OUTPUT="output/synthetic_crosspass_robustness"

if [[ -f "robustness_settings/settings.synthetic_robust.reference.ini" ]]; then
    SETTINGS_DIR="robustness_settings"
elif [[ -f "settings.synthetic_robust.reference.ini" ]]; then
    SETTINGS_DIR="."
else
    echo "Cannot find the robustness INI files." >&2
    echo "Place them in ./robustness_settings or next to this script." >&2
    exit 1
fi

CONFIGS=(
    "settings.synthetic_robust.reference.ini"
    "settings.synthetic_robust.scales_fine.ini"
    "settings.synthetic_robust.scales_coarse.ini"
    "settings.synthetic_robust.persistence_low.ini"
    "settings.synthetic_robust.persistence_high.ini"
    "settings.synthetic_robust.shoulder_strict.ini"
    "settings.synthetic_robust.shoulder_permissive.ini"
)

mkdir -p "$COMMON_OUTPUT"

for config in "${CONFIGS[@]}"; do
    echo
    echo "Running $config"
    PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" -m posterior_landscape "$SETTINGS_DIR/$config"
done

echo
echo "All runs completed. Outputs are in $COMMON_OUTPUT/"
