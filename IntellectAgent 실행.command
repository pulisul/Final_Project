#!/usr/bin/env bash
cd "$(dirname "$0")"
bash run.sh app
read -n 1 -s -r -p "종료하려면 아무 키나 누르세요"
