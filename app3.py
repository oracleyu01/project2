import streamlit as st
import os
import json
import numpy as np
from supabase import create_client
from openai import OpenAI

# 페이지 구성
st.set_page_config(page_title="네이버 통합 검색", layout="wide")

# Streamlit에서 실행 중인지 확인하고 secrets 가져오기
try:
    # Streamlit Cloud 환경에서는 st.secrets 사용
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    # 로컬 환경에서는 환경 변수 사용
    try:
        import dotenv
        dotenv.load_dotenv()
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    except:
        st.error("API 키를 가져오는 데 실패했습니다. 환경 변수나 Streamlit Secrets가 제대로 설정되었는지 확인하세요.")
        st.stop()

# API 키 확인
if not supabase_url or not supabase_key or not openai_api_key:
    st.error("필요한 API 키가 설정되지 않았습니다.")
    st.stop()

# Supabase 클라이언트 초기화
try:
    supabase = create_client(supabase_url, supabase_key)
    st.sidebar.success("Supabase 연결 성공!")
except Exception as e:
    st.error(f"Supabase 연결 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# OpenAI 클라이언트 초기화
try:
    openai_client = OpenAI(api_key=openai_api_key)
    st.sidebar.success("OpenAI 연결 성공!")
except Exception as e:
    st.error(f"OpenAI 연결 중 오류가 발생했습니다: {str(e)}")
    st.stop()

def generate_embedding(text):
    """텍스트에서 OpenAI 임베딩 생성"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"임베딩 생성 중 오류 발생: {str(e)}")
        raise

def semantic_search(query_text, source_type="블로그"):
    """시맨틱 검색 수행"""
    try:
        # 기본값 설정
        limit = 20  # 더 많은 결과를 먼저 가져옴 (필터링 후 최대 10개만 사용)
        match_threshold = 0.5  # 유사도 임계값 고정
        
        # 쿼리 텍스트에 대한 임베딩 생성
        query_embedding = generate_embedding(query_text)
        
        # 기존의 match_documents 함수 사용
        try:
            response = supabase.rpc(
                'match_documents', 
                {
                    'query_embedding': query_embedding,
                    'match_threshold': match_threshold,
                    'match_count': limit
                }
            ).execute()
            
            if response.data and len(response.data) > 0:
                # 클라이언트 측에서 소스 타입에 따라 필터링
                filtered_results = []
                for item in response.data:
                    # metadata 확인
                    metadata = item.get('metadata', {})
                    item_source_type = metadata.get('collection', '')
                    
                    # 소스 타입이 일치하는 경우에만 추가
                    if item_source_type == source_type:
                        filtered_results.append(item)
                
                # 최대 10개 결과만 반환
                return filtered_results[:10]
            else:
                return []
                
        except Exception as e:
            st.sidebar.warning(f"RPC 검색 실패: {str(e)}")
            return []
        
    except Exception as e:
        st.error(f"시맨틱 검색 중 오류 발생: {str(e)}")
        raise

def generate_answer_with_gpt(query, search_results, source_type):
    """GPT-4o-mini를 사용하여 검색 결과에 기반한 답변 생성"""
    try:
        # 검색 결과가 없는 경우
        if not search_results:
            return f"죄송합니다. 입력하신 '{query}'에 대한 {source_type} 검색 결과를 찾을 수 없습니다. 다른 검색어나 다른 소스 타입으로 시도해보세요."
            
        # 검색 결과를 컨텍스트로 정리
        contexts = []
        for i, result in enumerate(search_results[:5]):  # 상위 5개 결과만 사용
            content = result['content']
            metadata = result.get('metadata', {})
            title = metadata.get('title', '제목 없음')
            date = metadata.get('date', '')  # 날짜 정보가 있으면 추가
            
            # 날짜 정보가 있으면 포함
            date_info = f" (작성일: {date})" if date else ""
            
            # 출처 타입과 함께 컨텍스트 추가
            contexts.append(f"문서 {i+1} - [{source_type}] {title}{date_info}:\n{content}\n")
        
        context_text = "\n".join(contexts)
        
        # GPT 프롬프트 작성
        prompt = f"""다음은 네이버 {source_type}에서 수집한 데이터입니다:

{context_text}

위 내용을 바탕으로 다음 질문에 상세히 답변해주세요: 
"{query}"

답변 작성 규칙:
1. 한국어로 자연스럽게 답변해주세요.
2. 제공된 문서 내용만 사용하여 사실에 기반한 답변을 작성해주세요.
3. 문서에 없는 내용은 추측하거나 답변하지 마세요.
4. 여러 문서 간에 상충되는 정보가 있다면 이를 언급해주세요.
5. 답변에 적절한 정보가 부족하다면 솔직하게 말씀해주세요.
6. 답변은 논리적인 구조로 정리하여 사용자가 이해하기 쉽게 작성해주세요.
7. 필요한 경우 정보의 출처를 언급해주세요(예: "문서 2에 따르면...").
"""

        # GPT-4o-mini로 답변 생성
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"당신은 네이버 {source_type} 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다. 사용자 질문에 맞는 최적의 답변을 제공하세요. 문서에 없는 내용은 추가하지 말고 정확한 사실만 전달하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 일관성 있는 답변을 위해 낮은 온도 설정
            max_tokens=1000   # 충분한 답변 길이
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"GPT 답변 생성 중 오류 발생: {str(e)}")
        return "답변 생성 중 오류가 발생했습니다."

# 메인 UI
st.title("네이버 통합 검색 & 질의응답")
st.write("Supabase 벡터 데이터베이스에 저장된 네이버 데이터를 검색하고 질문에 답변합니다.")

# 검색 소스 선택 (라디오 버튼)
col1, col2, col3 = st.columns(3)
with col1:
    blog_selected = st.radio("검색 소스 선택", options=["블로그", "뉴스", "쇼핑"], index=0)

# 검색 입력
query = st.text_input("질문 입력", value="안성탕면 맛있게 끓이는 방법이 뭐지?", help=f"네이버 {blog_selected}에 관한 질문을 입력하세요")

# 원본 검색 결과 표시 옵션
show_raw_results = st.sidebar.checkbox("원본 검색 결과 표시", value=True)

# 검색 버튼
if st.button(f"{blog_selected} 검색", key="search_button"):
    if query:
        with st.spinner(f"{blog_selected} 검색 중..."):
            try:
                # 선택한 소스 타입으로 시맨틱 검색 수행
                results = semantic_search(query, source_type=blog_selected)
                
                if results:
                    st.success(f"{len(results)}개의 {blog_selected} 결과를 찾았습니다.")
                    
                    # GPT로 답변 생성
                    with st.spinner("AI 에이전트 답변 생성 중..."):
                        gpt_answer = generate_answer_with_gpt(query, results, blog_selected)
                        
                        # 답변 표시
                        st.markdown(f"## AI 답변 ({blog_selected} 데이터 기반)")
                        st.markdown(gpt_answer)
                        
                        # 구분선
                        st.markdown("---")
                    
                    # 원본 검색 결과 표시 옵션
                    if show_raw_results:
                        st.markdown(f"## {blog_selected} 검색 결과 원본")
                        for i, result in enumerate(results):
                            similarity = result['similarity'] * 100  # 백분율로 변환
                            
                            # 메타데이터에서 정보 추출
                            metadata = result.get('metadata', {})
                            title = metadata.get('title', '제목 없음')
                            
                            # URL 추출
                            url = None
                            if 'url' in metadata:
                                url = metadata['url']
                            
                            # 결과 표시
                            with st.expander(f"{i+1}. {title} (유사도: {similarity:.2f}%)"):
                                st.write(f"**내용:** {result['content'][:300]}...")
                                
                                # 메타데이터 정보 표시
                                meta_col1, meta_col2 = st.columns(2)
                                
                                with meta_col1:
                                    if blog_selected == "블로그" and 'bloggername' in metadata:
                                        st.write(f"**블로거:** {metadata['bloggername']}")
                                    elif blog_selected == "뉴스" and 'publisher' in metadata:
                                        st.write(f"**언론사:** {metadata['publisher']}")
                                    elif blog_selected == "쇼핑" and 'maker' in metadata:
                                        st.write(f"**제조사:** {metadata['maker']}")
                                        
                                    if 'date' in metadata:
                                        st.write(f"**날짜:** {metadata['date']}")
                                
                                with meta_col2:
                                    if url:
                                        st.markdown(f"**링크:** [원본 {blog_selected} 보기]({url})")
                                    if blog_selected == "쇼핑" and 'price' in metadata:
                                        st.write(f"**가격:** {metadata['price']}원")
                else:
                    st.warning(f"{blog_selected}에서 검색 결과가 없습니다. 다른 검색어나 다른 소스 타입으로 시도해보세요.")
            
            except Exception as e:
                st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
    else:
        st.warning("질문을 입력하세요.")

# 데이터베이스 상태
st.sidebar.title("데이터베이스 상태")
try:
    # 전체 문서 수 가져오기
    result = supabase.table('documents').select('id', count='exact').execute()
    doc_count = result.count if hasattr(result, 'count') else len(result.data)
    st.sidebar.info(f"저장된 문서 수: {doc_count}개")
except Exception as e:
    st.sidebar.error("데이터베이스 상태를 확인할 수 없습니다.")

# 사용 안내
st.sidebar.title("사용 안내")
st.sidebar.info(f"""
1. 검색 소스 선택: 블로그, 뉴스, 쇼핑 중에서 검색할 소스를 선택하세요.
2. 질문 입력: 검색하고자 하는 내용을 입력하세요.
3. '{blog_selected} 검색' 버튼을 클릭하면 AI가 관련 정보를 찾아 답변합니다.
4. 원본 검색 결과 표시: 체크하면 AI 답변 아래에 원본 검색 결과도 함께 표시됩니다.
""")
