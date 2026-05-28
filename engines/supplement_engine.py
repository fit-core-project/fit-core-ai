import os
import time
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# LangChain & AI 관련 임포트
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from engines.log_redaction import sanitize_exception_for_log
from engines.llm_router import get_llm

# 🌐 웹 검색 툴 추가
from langchain_community.tools import DuckDuckGoSearchRun

# BM25 & Reranker
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv

load_dotenv()

class SupplementRAGEngine:
    def __init__(self, db_path: str = "./data/chroma_db"):
        print("💊 [Agentic Supplement RAG] 엔진 초기화 시작...")
        self.device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

        # 1. 생성용 LLM 세팅 (gemini-2.5-flash)
        self.llm = get_llm("supplement", temperature=0.1)
        self.query_optimizer = get_llm("supplement", temperature=0.0)

        # 🌐 웹 검색기 초기화
        self.web_search = DuckDuckGoSearchRun()

        # 2. 벡터 DB 로드 (생략 없이 기존과 동일)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="dragonkue/BGE-m3-ko",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.db_dir = self._find_latest_db(db_path)
        if not self.db_dir:
            raise Exception("Chroma DB를 찾을 수 없습니다.")

        self.vector_store = Chroma(persist_directory=str(self.db_dir), embedding_function=self.embeddings)

        # 3. BM25 초기화
        self.kiwi = Kiwi(num_workers=-1)
        self.target_tags = ('N', 'V', 'X', 'S', 'VA')
        all_db_data = self.vector_store.get()
        self.original_docs = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(all_db_data["documents"], all_db_data["metadatas"])
        ]
        clean_texts = [(d.page_content or "").strip() for d in self.original_docs]
        tokenized_corpus = [self._tokenize_kiwi(t) for t in clean_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 4. 리랭커 로드
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=self.device, max_length=512)

        # 5-A. 📝 로컬 DB용 프롬프트 (암호 지시 추가)
        self.local_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공인된 전문 약사이자 헬스케어 AI 코치입니다.
아래 [검증된 문서]를 읽고 질문에 답하세요.

[매우 중요한 지시사항]
만약 문서 안에 질문과 관련된 내용이나 상호작용 정보가 **전혀 없다면**, 절대 억지로 지어내거나 일반적인 조언을 하지 말고, 오직 "WEB_SEARCH_REQUIRED" 라고만 출력하세요.

답변이 가능하다면 다음 형식을 지키세요:
1. 답변 요약 (핵심 내용)
2. 주의사항 (위험 요소 강조)
3. "자세한 내용은 전문가와 상담하세요." 추가

[검증된 문서]
{context}
"""),
            ("human", "{question}")
        ])

        # 5-B. 🌐 웹 검색용 프롬프트 (Fallback 전용)
        self.web_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 최신 의학 정보를 바탕으로 조언하는 전문 약사입니다.
로컬 데이터베이스에 정보가 없어, 방금 실시간 웹 검색을 수행했습니다. 
아래 [웹 검색 결과]를 분석하여 사용자에게 안전하고 친절하게 답변해 주세요.

[웹 검색 결과]
{context}
"""),
            ("human", "{question}")
        ])

    def _find_latest_db(self, base_path: str) -> Path:
        base_dir = Path(base_path)
        fixed_path = base_dir / "latest_index"
        if fixed_path.exists(): return fixed_path
        if not base_dir.exists(): return None
        db_folders = sorted(list(base_dir.glob("doc_index_*")), reverse=True)
        return db_folders[0] if db_folders else None

    def _tokenize_kiwi(self, text: str):
        if not text: return []
        return [t.form for t in self.kiwi.tokenize(text) if t.tag.startswith(self.target_tags)]

    def _expand_query(self, query: str) -> str:
        prompt = f"질문에서 '성분명/질환명/증상/약물군' 핵심 키워드만 추출해. 공백으로 구분. 부연설명 절대 금지.\n질문: {query}"
        try: return self.query_optimizer.invoke(prompt).content.strip() or query
        except: return query

    def _doc_key(self, doc: Document):
        meta = doc.metadata or {}
        return meta.get("id") or meta.get("_source_file") or hash(doc.page_content)

    def answer_question(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            search_query = self._expand_query(question)

            # Step 1: 하이브리드 검색 및 RRF
            k = 30
            vector_docs = self.vector_store.similarity_search(search_query, k=k)

            q_tokens = self._tokenize_kiwi(search_query)
            bm25_docs = []
            if q_tokens:
                scores = self.bm25.get_scores(q_tokens)
                top_idx = np.argsort(scores)[::-1][:k]
                bm25_docs = [self.original_docs[i] for i in top_idx]

            rrf_k = 60
            fused_scores = {}
            doc_map = {}

            def add_to_rrf(docs, weight):
                for rank, doc in enumerate(docs):
                    key = self._doc_key(doc)
                    if key not in fused_scores:
                        fused_scores[key] = 0.0
                        doc_map[key] = doc
                    fused_scores[key] += weight * (1.0 / (rank + rrf_k))

            add_to_rrf(bm25_docs, 0.6)
            add_to_rrf(vector_docs, 0.4)

            ranked_keys = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)[:15]
            candidate_docs = [doc_map[k] for k in ranked_keys]

            # Step 2: Cross-Encoder 리랭킹 (시야를 넓혀 8개 참조)
            pairs = [[question, doc.page_content] for doc in candidate_docs]
            rerank_scores = self.reranker.predict(pairs)
            reranked_results = sorted(zip(rerank_scores, candidate_docs), key=lambda x: x[0], reverse=True)
            final_docs = [doc for score, doc in reranked_results[:8]]

            context_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])

            # Step 3: 로컬 DB 기반 답변 시도
            local_chain = self.local_prompt | self.llm | StrOutputParser()
            answer = local_chain.invoke({"context": context_text, "question": question})

            # 🌐 Step 4: 웹 검색 폴백 (Agentic Action)
            if "WEB_SEARCH_REQUIRED" in answer.strip().upper():
                print("🌐 [Agent] 로컬 DB에 정보가 부족합니다. 실시간 웹 검색을 시작합니다...")

                # 검색어에 '약학', '상호작용' 등을 붙여 검색 품질을 높임
                web_query = f"{search_query} 복용법 약효 부작용"
                web_result_text = self.web_search.run(web_query)

                web_chain = self.web_prompt | self.llm | StrOutputParser()
                answer = web_chain.invoke({"context": web_result_text, "question": question})

                sources = ["🌐 실시간 웹 검색 데이터 (DuckDuckGo)"]
            else:
                sources = []
                seen_keys = set()

                for doc in final_docs:
                    meta = doc.metadata
                    # 파일명과 ID를 조합해서 중복 출처 제거
                    source_key = f"{meta.get('_source_file')}_{meta.get('id')}"

                    if source_key not in seen_keys:
                        sources.append({
                            "file": meta.get('_source_file', '알 수 없는 파일'),
                            "id": str(meta.get('id', 'N/A')),
                            "type": str(meta.get('source', 'DOC')) # drugs, supp, faq 등
                        })
                        seen_keys.add(source_key)

            print(f"✅ 총 소요 시간: {time.time() - start_time:.2f}초")

            # 최종 리턴: 이제 문자열 리스트가 아니라 '객체 리스트'를 보냅니다.
            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            # 에러 발생 시 처리
            print("❌ [Advanced RAG 에러]:", sanitize_exception_for_log(e))
            return {"answer": "오류가 발생했습니다.", "sources": []}
