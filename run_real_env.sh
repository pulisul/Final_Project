#!/usr/bin/env bash
# 예전 명령(setup/docs/qa/eval/app)을 그대로 쓰는 사람을 위한 연결 스크립트. 실제 일은 install.sh / run.sh 가 한다.
cd "$(dirname "${BASH_SOURCE[0]}")"
case "${1:-}" in
  setup) shift; exec bash install.sh --yes "$@";;
  "")    exec bash run.sh --help;;
  *)     exec bash run.sh "$@";;
esac
