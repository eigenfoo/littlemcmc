#!/bin/bash
# Checks if there have been any commits to pymc3/step_methods/ since yesterday.

set -e

git clone https://github.com/pymc-devs/pymc3.git
cd pymc3/
num_commits=$(git log --since="1 day ago" pymc3/step_methods/hmc/ pymc3/sampling.py | grep "^commit [a-z0-9]*$" | wc -l)

if [ "$num_commits" -eq "0" ]; then
    echo "No commits since yesterday. Passing."
    exit 0
else
    echo "$num_commits commits since yesterday. Failing."
    echo ""
    git log --since="1 day ago" pymc3/step_methods/hmc/ pymc3/sampling.py | grep "^commit [a-z0-9]*$" | sed "s/commit /https:\/\/github.com\/pymc-devs\/pymc3\/commit\//"
    exit 1
fi
