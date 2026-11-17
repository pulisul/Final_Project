#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from intellectagent.hashing import sha256_file

DOCS=ROOT/'data/sample_docs'
OUT=ROOT/'data/eval/week6/eval20.jsonl'

def doc_id(name:str)->str:
    return sha256_file(DOCS/name)[:16]

rag=doc_id('rag.md'); apples=doc_id('apples.md')
rows=[
('q01','RAG는 무엇인가요?', [rag], 'RAG는 질문과 관련된 문서를 검색한 뒤 검색 근거를 생성 단계에 제공하는 구조입니다.',False,'fact'),
('q02','RAG에서 먼저 수행하는 작업은 무엇인가요?', [rag], '질문과 관련된 문서를 먼저 검색합니다.',False,'fact'),
('q03','검색된 근거는 어디에 제공되나요?', [rag], '검색된 근거는 생성 단계에 제공됩니다.',False,'fact'),
('q04','이 프로젝트의 핵심 게이트가 요구하는 것은 무엇인가요?', [rag], '검색 결과와 출처가 연결되어야 합니다.',False,'fact'),
('q05','검색 결과와 출처의 관계는 무엇인가요?', [rag], '검색 결과와 출처가 연결되어야 합니다.',False,'fact'),
('q06','사과는 무엇인가요?', [apples], '사과는 과일입니다.',False,'fact'),
('q07','사과에는 여러 품종이 있나요?', [apples], '사과에는 여러 품종이 있습니다.',False,'fact'),
('q08','사과 샘플 문서는 무엇을 검증하나요?', [apples], 'RAG 검색과 출처 검증의 재현성 테스트에 사용됩니다.',False,'fact'),
('q09','샘플 문서에서 과일로 언급된 것은 무엇인가요?', [apples], '사과입니다.',False,'fact'),
('q10','RAG 문서와 사과 문서는 어떤 테스트에 쓰이나요?', [rag,apples], 'RAG 구조와 검색·출처 검증의 재현성을 시험하는 샘플입니다.',False,'multi_doc'),
('q11','두 샘플 문서가 공통으로 지원하는 프로젝트 목적은 무엇인가요?', [rag,apples], 'RAG 검색과 출처 연결을 재현 가능하게 검증하는 목적입니다.',False,'multi_doc'),
('q12','문서 검색과 생성은 어떤 순서로 연결되나요?', [rag], '문서를 먼저 검색하고 검색 근거를 생성 단계에 제공합니다.',False,'fact'),
('q13','출처 검증 재현성 테스트용 문서는 무엇인가요?', [apples], '사과 샘플 문서입니다.',False,'fact'),
('q14','RAG에서 생성 단계가 받는 입력 중 문서와 관련된 것은 무엇인가요?', [rag], '검색된 근거입니다.',False,'fact'),
('q15','검색 결과와 출처가 연결되지 않으면 핵심 게이트를 통과하나요?', [rag], '통과하지 않습니다.',False,'fact'),
('q16','화성 기지 건설 예산은 얼마인가요?', [], '문서에 근거가 없습니다.',True,'fallback'),
('q17','2028년 서울의 정확한 인구는 얼마인가요?', [], '문서에 근거가 없습니다.',True,'fallback'),
('q18','양자컴퓨터의 최신 시장 점유율은 얼마인가요?', [], '문서에 근거가 없습니다.',True,'fallback'),
('q19','특정 회사의 최신 매출을 알려주세요.', [], '문서에 근거가 없습니다.',True,'fallback'),
('q20','이 문서에 없는 해외 주식의 내일 종가는 얼마인가요?', [], '문서에 근거가 없습니다.',True,'fallback'),
]
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('w',encoding='utf-8') as f:
    for cid,q,docs,ref,fb,qt in rows:
        f.write(json.dumps({'case_id':cid,'question':q,'relevant_doc_ids':docs,'reference_answer':ref,'should_fallback':fb,'question_type':qt},ensure_ascii=False)+'\n')
print(OUT)
