#!/bin/bash

cd /home/dkleis/dev/tischkicker || exit

git fetch origin live

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/live)

if [ "$LOCAL" != "$REMOTE" ]; then
    echo "$(date): Update gefunden, ziehe neuen Stand"
    git pull origin live
else
    echo "$(date): Kein Update"
fi
