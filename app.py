import streamlit as st
import json
import os
import tempfile
from datetime import datetime
from supabase import create_client
import numpy as np
from openai import OpenAI
import dotenv

# 환경 변수 로드
dotenv.load_dotenv()

# Supabase 클라이언트 초기화
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# OpenAI 클라이언트 초기화 (벡터 임베딩용)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_embedding(text):
    """텍스트에서 OpenAI 임베딩 생성"""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def process_json_file(file_path, collection_name=None):
    """JSON 파일 처리 및 Supabase에 저장"""
    # JSON 파일 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 네이버 API 응답 구조 확인
    if isinstance(data, dict) and 'items' in data:
        # 네이버 API 응답 형식인 경우
        items = data['items']
    else:
        # 직접 JSON 배열인 경우
        items = data

    # 컬렉션 이름 생성
    if not collection_name:
        keyword_safe = "realstate"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection_name = f'{keyword_safe}_{timestamp}'

    # 처리된 문서 수 카운트
    doc_count = 0

    # 각 항목 처리
    for i, item in enumerate(items):
        # 필드 확인 및 대체
        title = item.get('title', '')
        if '<b>' in title:  # HTML 태그 제거
            title = title.replace('<b>', '').replace('</b>', '')

        # 설명/내용 필드 (네이버 API는 'description'을 사용)
        content = item.get('description', item.get('content', ''))
        if '<b>' in content:
            content = content.replace('<b>', '').replace('</b>', '')

        # 글의 전체 내용 (제목 + 내용)
        full_content = title + " " + content

        # 임베딩 생성
        embedding = generate_embedding(full_content)

        # 메타데이터 구성
        metadata = {
            "title": title,
            "collection": collection_name,
            "collected_at": datetime.now().isoformat()
        }

        # 네이버 API 필드들 추가
        if 'link' in item:
            metadata['url'] = item['link']

        if 'pubDate' in item:  # 뉴스의 경우
            metadata['date'] = item['pubDate']
        elif 'postdate' in item:  # 블로그의 경우
            metadata['date'] = item['postdate']

        if 'bloggername' in item:
            metadata['bloggername'] = item['bloggername']

        # Supabase에 데이터 삽입
        data = {
            'content': full_content,
            'embedding': embedding,
            'metadata': metadata
        }

        supabase.table('documents').insert(data).execute()
        doc_count += 1

    return collection_name, doc_count

# Streamlit 앱 UI
st.title("네이버 JSON 파일을 Supabase에 저장하기")

uploaded_file = st.file_uploader("JSON 파일 업로드", type=['json'])

if uploaded_file is not None:
    # 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # 컬렉션 이름 입력
    collection_name = st.text_input("컬렉션 이름 (입력하지 않으면 자동 생성됩니다)")

    if st.button("Supabase에 저장"):
        with st.spinner("데이터 처리 중..."):
            try:
                collection_name, doc_count = process_json_file(tmp_file_path, collection_name)
                st.success(f"성공적으로 {doc_count}개의 문서가 저장되었습니다!")
                st.write(f"컬렉션 이름: {collection_name}")
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
            finally:
                # 임시 파일 삭제
                os.unlink(tmp_file_path)
