#!/bin/bash

FILE=$1
SPLIT=$2

[[ $SPLIT == val ]] && NUM=11730 && DEN=15872
[[ $SPLIT == test ]] && NUM=224185 && DEN=301392

echo $NUM
echo $DEN

cat $FILE | grep "\- Eval Score" | cut -d ' ' -f10 | while IFS="$IFS," read x; do printf '%f\n' "$((x * $NUM / $DEN))"; done
