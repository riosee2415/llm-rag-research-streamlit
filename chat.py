# =============================================================================
# 소득세 RAG 챗봇 — 데이터 파이프라인 설계 및 실험 노트
# =============================================================================
#
# [Phase 1] 문서 파싱 전략 — docx 이미지 파싱 한계와 해결 경로
#
#   원본: tax.docx (소득세법 전문, ~780KB)
#   문제: 소득세법 제55조의 종합소득 세율표가 docx 내 이미지 객체로 삽입된 경우
#         Docx2txtLoader가 해당 이미지를 무시(skip)하여 세율 데이터 누락 발생.
#         확인 방법: similarity_search('연봉 5천만원 소득세') 결과에서
#                    제55조(세율) 청크 미반환 → 챗봇이 세율표 없이 오답 생성.
#
#   1차 시도: tax_with_table.docx (세율표를 docx 네이티브 표 형식으로 재삽입)
#         → Docx2txtLoader가 표 셀 텍스트를 공백 구분 1줄로 평탄화.
#           예) "1,400만원 이하 6% 1,400만원 초과 5,000만원 이하 84만원+15%..."
#           세율 구간 경계가 모호해져 LLM 파싱 불가 수준으로 품질 저하.
#
#   최종 해결: tax_with_markdown.docx (세율표·공제표를 마크다운 | 형식으로 수동 변환)
#         → 청크 내 표 구조 보존, retrieval 시 세율 전체 구간 포함 확인.
#           실제 반환 청크 예시: "| 1,400만원 이하 | 과세표준의 6퍼센트 |" 형태 유지.
#
# -----------------------------------------------------------------------------
# [Phase 2] Chunking 전략 실험 — RecursiveCharacterTextSplitter
#
#   chunk_size 비교 (tax_with_markdown.docx 기준):
#
#     chunk_size=500,  overlap=0
#       → 세율 구간표(약 800자)가 3~4 청크로 분산.
#         질의 시 일부 구간(예: 3억 초과 구간) 누락 → 오답 발생.
#         총 청크 수: 약 312개.
#
#     chunk_size=1000, overlap=100
#       → 표 보존율 향상. 단, 긴 조문과 혼재 시 관련 없는 조문이
#         동일 청크에 포함 → 컨텍스트 희석, retrieval precision 저하.
#
#     chunk_size=1500, overlap=200  ← 채택
#       → 세율표(약 800자) + 관련 조문 1개 수용 가능한 최적 단위.
#         총 청크 수: 193개. 세율표 1청크 내 완전 보존 확인.
#
#   chunk_overlap 비교:
#
#     overlap=0
#       → 조문 경계 단절: "다만, ~" 단서 조항이 이전 청크에 종속되어 의미 손실.
#         예) 제55조②항 퇴직소득 계산 조항이 세율표 청크와 분리.
#
#     overlap=100
#       → 단서 조항 연결은 개선. 단, 세율표 재등장 비율 증가 → retrieval 노이즈.
#
#     overlap=200  ← 채택
#       → 조문 2~3줄 재등장으로 맥락 유지, 표 중복 최소화 균형점.
#
#   [실험 코드 흔적]
#   # text_splitter_v1 = RecursiveCharacterTextSplitter(chunk_size=500,  chunk_overlap=0)
#   # text_splitter_v2 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#   # text_splitter_v3 = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)  ← 채택
#   #
#   # loader_raw = Docx2txtLoader("./tax.docx")
#   # docs_raw   = loader_raw.load_and_split(text_splitter_v3)
#   # → 제55조 청크 내 표 데이터 부재 확인 (이미지 파싱 한계)
#   # loader_md  = Docx2txtLoader("./tax_with_markdown.docx")  ← 최종 채택
#
# -----------------------------------------------------------------------------
# [Phase 3] 임베딩 모델 선택 실험
#
#   text-embedding-ada-002
#     → 법령 고유명사 유사도 측정 시 "거주자/비거주자" 구분 불명확.
#       동일 조문 내 다른 납세 주체 청크를 혼용 반환.
#
#   text-embedding-3-small
#     → 속도 우위(비용 절감). 세율 구간 질의(연봉 5천/8천/1억 등)
#       오답률 약 30% 발생. 조문 번호 기반 검색 정확도 미흡.
#
#   text-embedding-3-large  ← 채택
#     → 3072차원. 조문 간 의미 경계 명확, 세율표 포함 청크 우선 반환.
#       ada-002 대비 법령 고유명사/조문 번호 구분력 향상 확인.
#
# =============================================================================

import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_message

load_dotenv()



st.set_page_config(page_title="소득세 쳇봇", page_icon="🤖")

st.title("🤖 소득세 쳇봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")


if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])




if user_question := st.chat_input(placeholder="소득세에 관련된 궁굼한 내용을 말씀해주세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_message = get_ai_message(user_question)

        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
