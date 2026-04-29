from dotenv import load_dotenv


# LLM + RAG packages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_classic import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore




load_dotenv()

def get_ai_message(user_message):
    # [실험 결과] text-embedding-3-large 채택 이유:
    # ada-002 대비 법령 고유명사/조문 번호 구분력 향상.
    # 3-small 대비 세율 구간 질의(연봉 5천/8천/1억) 정답률 개선 확인.
    embedding = OpenAIEmbeddings(model = 'text-embedding-3-large')
    index_name = 'tax-table-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

    llm = ChatOpenAI(model='gpt-4o')
    prompt = hub.pull('rlm/rag-prompt')

    # [k 값 실험]
    # k=1: 세율표 단독 반환 → 공제·납세지 복합 질의 실패.
    # k=3: 초기 기본값, 단순 세율 질의는 충분 (노트북 1차 실험값).
    # k=4: 채택 — 세율표(1청크) + 관련 조문(2~3청크) 조합으로 복합 질의 응답 품질 최적.
    retriever=database.as_retriever(search_kwargs={'k': 4})

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={'prompt': prompt})

    # [법령 용어 정규화 필요성]
    # 소득세법은 납세 주체를 반드시 "거주자/비거주자"로 구분 (소득세법 제1조의2).
    # 실험: "직장인의 소득세는?" 원문 질의 → 임베딩 공간에서 "거주자" 벡터와 거리 발생.
    #        → 제55조(세율) 청크 미반환 → 오답 또는 "모르겠습니다" 응답 확인.
    # 해결: LLM 기반 전처리 체인으로 구어체 표현을 법령 용어로 변환 후 검색 수행.
    # 검증: "직장인" → "거주자" 변환 후 동일 질의 재실행 → 세율표 청크 정상 반환.
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단되다면, 사용자의 질문을 변경하지 않아도 됩니다.
        변경하지 않아도 되는 경우에는 질문만 리턴해주세요.

        사전 : {dictionary}

        질문 : {{question}}
    """)

    # [2-Stage Chain 설계]
    # Stage 1 - dictionary_chain: 사용자 질문 → 법령 용어로 정규화 (LLM 전처리).
    # Stage 2 - qa_chain: 정규화된 질문으로 Pinecone 벡터 검색 + 답변 생성.
    # 단일 chain 대비 법령 특화 질의에서 retrieval precision 향상 목적.
    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {"query" : dictionary_chain} | qa_chain
    ai_response = tax_chain.invoke({"question" : user_message})

    return ai_response["result"]
