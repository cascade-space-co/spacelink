#!/usr/bin/env bash
# Derives the package version from git tags of the form "vX.Y.Z".
#
# On an exact tag: prints the tag with the leading "v" stripped.
# Otherwise: prints "<last-tag>.dev<commits-since>+g<short-hash>", a PEP
# 440 dev-release that sorts between the last tag and the next one.
#
# --strict: require HEAD to have an exact tag matching vX.Y.Z
set -euo pipefail

version_re='^v[0-9]+\.[0-9]+\.[0-9]+$'
strict=false
[[ "${1:-}" == "--strict" ]] && strict=true

if tag=$(git describe --tags --match 'v*' --exact-match 2>/dev/null); then
    if $strict && [[ ! "$tag" =~ $version_re ]]; then
        echo "error: tag '$tag' on HEAD does not match the required vX.Y.Z format" >&2
        exit 1
    fi
    echo "${tag#v}"
    exit 0
fi

if $strict; then
    found=$(git describe --tags --exact-match 2>/dev/null || echo "<none>")
    echo "error: HEAD has no exact vX.Y.Z tag (found: $found)" >&2
    exit 1
fi

describe=$(git describe --tags --match 'v*' --long)
tag=${describe%-*-*}
rest=${describe#"$tag"-}
n=${rest%-g*}
hash=${rest##*-g}
echo "${tag#v}.dev${n}+g${hash}"
