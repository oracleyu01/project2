import streamlit as st
import os
import json
import numpy as np
import urllib.request
import urllib.parse
import re
import pandas as pd
from datetime import datetime
from supabase import create_client
from openai import OpenAI
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud

# 페이지 구성
st.set_page_config(page_title="네이버 통합 검색", layout="wide")

# 네이버 API 클라이언트 ID와 시크릿
NAVER_CLIENT_ID = "9XhhxLV1IzDpTZagoBr1"
NAVER_CLIENT_SECRET = "J14HFxv3B6"

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

def visualize_search_results(results, source_type):
    """시맨틱 검색 결과를 시각화"""
    if not results:
        return None
    
    # 시각화 컨테이너
    visualizations = {}
    
    # 1. 유사도 분포 그래프
    similarities = [result['similarity'] * 100 for result in results]
    
    fig_similarity = go.Figure(data=[
        go.Bar(x=list(range(1, len(similarities) + 1)), 
               y=similarities,
               marker_color='skyblue')
    ])
    fig_similarity.update_layout(
        title=f"{source_type} 검색 결과 유사도 분포",
        xaxis_title="검색 결과 순위",
        yaxis_title="유사도 (%)",
        yaxis_range=[0, 100]
    )
    visualizations['similarity'] = fig_similarity
    
    # 2. 키워드 빈도 분석 (워드 클라우드)
    all_text = " ".join([result['content'] for result in results])
    
    # 불용어 목록 (필요에 따라 확장)
    stopwords = ['있는', '없는', '그리고', '그런', '하는', '이런', '있다', '때문에', '그것은', '이것은', '그냥', '정말', '매우']
    
    # 형태소 분석 대신 간단한 단어 추출 (2글자 이상 단어만)
    words = re.findall(r'\w{2,}', all_text)
    words = [word for word in words if word not in stopwords]
    word_counts = Counter(words)
    
    # 가장 많이 나타나는 상위 15개 단어
    top_words = dict(word_counts.most_common(15))
    
    # 워드 클라우드 생성
    try:
        wordcloud = WordCloud(
            font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # 한글 폰트 경로
            width=800, 
            height=400, 
            background_color='white'
        ).generate_from_frequencies(top_words)
        
        # matplotlib 그림 생성 
        fig_wordcloud = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        visualizations['wordcloud'] = fig_wordcloud
    except Exception as e:
        st.warning(f"워드 클라우드 생성 오류: {str(e)}")
        # 워드 클라우드가 실패하면 막대 그래프로 대체
        fig_words = go.Figure(data=[
            go.Bar(
                x=list(top_words.keys()),
                y=list(top_words.values()),
                marker_color='lightgreen'
            )
        ])
        fig_words.update_layout(
            title=f"{source_type} 검색 결과 주요 키워드",
            xaxis_title="키워드",
            yaxis_title="빈도수"
        )
        visualizations['word_freq'] = fig_words
    
    # 3. 소스 타입별 추가 시각화
    if source_type == "블로그":
        # 블로거별 결과 수
        bloggers = []
        for result in results:
            metadata = result.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    continue
            
            blogger = metadata.get('bloggername', '알 수 없음')
            bloggers.append(blogger)
        
        # 블로거 분포 그래프
        blogger_counts = Counter(bloggers)
        fig_bloggers = go.Figure(data=[
            go.Pie(
                labels=list(blogger_counts.keys()),
                values=list(blogger_counts.values()),
                hole=.3
            )
        ])
        fig_bloggers.update_layout(title="블로거별 검색 결과 분포")
        visualizations['blogger_dist'] = fig_bloggers
        
    elif source_type == "뉴스":
        # 언론사별 결과 수
        publishers = []
        for result in results:
            metadata = result.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    continue
            
            publisher = metadata.get('publisher', '알 수 없음')
            publishers.append(publisher)
        
        # 언론사 분포 그래프
        publisher_counts = Counter(publishers)
        fig_publishers = go.Figure(data=[
            go.Pie(
                labels=list(publisher_counts.keys()),
                values=list(publisher_counts.values()),
                hole=.3
            )
        ])
        fig_publishers.update_layout(title="언론사별 검색 결과 분포")
        visualizations['publisher_dist'] = fig_publishers
        
    elif source_type == "쇼핑":
        # 가격대 분포 분석
        price_ranges = {'~1만원': 0, '1~5만원': 0, '5~10만원': 0, '10~50만원': 0, '50만원~': 0}
        for result in results:
            metadata = result.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    continue
            
            price_str = metadata.get('lprice', '0')
            try:
                price = int(price_str)
                if price < 10000:
                    price_ranges['~1만원'] += 1
                elif price < 50000:
                    price_ranges['1~5만원'] += 1
                elif price < 100000:
                    price_ranges['5~10만원'] += 1
                elif price < 500000:
                    price_ranges['10~50만원'] += 1
                else:
                    price_ranges['50만원~'] += 1
            except:
                pass
        
        # 가격대 분포 그래프
        fig_prices = go.Figure(data=[
            go.Bar(
                x=list(price_ranges.keys()),
                y=list(price_ranges.values()),
                marker_color='salmon'
            )
        ])
        fig_prices.update_layout(title="가격대별 상품 분포")
        visualizations['price_dist'] = fig_prices
    
    return visualizations

def search_naver_api(query, source_type, count=20):
    """네이버 API를 사용하여 검색하고 결과를 Supabase에 저장"""
    try:
        # 소스 타입에 따른 API 엔드포인트 설정
        if source_type == "블로그":
            api_endpoint = "blog"
        elif source_type == "뉴스":
            api_endpoint = "news"
        elif source_type == "쇼핑":
            api_endpoint = "shop"
        else:
            api_endpoint = "blog"  # 기본값
        
        # 쿼리 인코딩
        encoded_query = urllib.parse.quote(query)
        url = f"https://openapi.naver.com/v1/search/{api_endpoint}?query={encoded_query}&display={count}&sort=sim"
        
        # 요청 헤더 설정
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
        request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
        
        # API 요청 및 응답 처리
        response = urllib.request.urlopen(request)
        response_code = response.getcode()
        
        if response_code == 200:
            # 응답 읽기 및 파싱
            response_body = response.read()
            response_data = json.loads(response_body.decode('utf-8'))
            
            # 결과 처리 및 Supabase에 저장
            saved_count = 0
            for i, item in enumerate(response_data.get('items', [])):
                # HTML 태그 제거
                title = re.sub('<[^<]+?>', '', item.get('title', ''))
                
                # 소스 타입에 따른 내용 필드 추출
                if source_type == "블로그":
                    content = re.sub('<[^<]+?>', '', item.get('description', ''))
                    metadata = {
                        'title': title,
                        'url': item.get('link', ''),
                        'bloggername': item.get('bloggername', ''),
                        'date': item.get('postdate', ''),
                        'collection': source_type
                    }
                elif source_type == "뉴스":
                    content = re.sub('<[^<]+?>', '', item.get('description', ''))
                    metadata = {
                        'title': title,
                        'url': item.get('link', ''),
                        'publisher': item.get('publisher', ''),
                        'date': item.get('pubDate', ''),
                        'collection': source_type
                    }
                elif source_type == "쇼핑":
                    content = f"{title}. " + re.sub('<[^<]+?>', '', item.get('category3', ''))
                    metadata = {
                        'title': title,
                        'url': item.get('link', ''),
                        'lprice': item.get('lprice', ''),
                        'hprice': item.get('hprice', ''),
                        'mallname': item.get('mallName', ''),
                        'maker': item.get('maker', ''),
                        'brand': item.get('brand', ''),
                        'collection': source_type
                    }
                
                # 전체 텍스트 생성 (임베딩용)
                full_text = f"{title} {content}"
                
                try:
                    # 임베딩 생성
                    embedding = generate_embedding(full_text)
                    
                    # Supabase에 데이터 삽입
                    data = {
                        'content': full_text,
                        'embedding': embedding,
                        'metadata': metadata
                    }
                    
                    # 이미 존재하는지 확인
                    # (URL 기반으로 중복 체크, 쇼핑은 상품 ID와 몰 이름으로 중복 체크)
                    check_field = 'url'
                    check_value = metadata.get('url', '')
                    
                    if source_type == "쇼핑" and 'productId' in item:
                        check_field = 'productId'
                        check_value = item.get('productId', '')
                    
                    # 중복 체크 쿼리 (메타데이터 내 필드 체크)
                    existing = supabase.table('documents').select('id').eq(f"metadata->{check_field}", check_value).execute()
                    
                    if not existing.data:  # 중복이 없을 경우에만 삽입
                        result = supabase.table('documents').insert(data).execute()
                        saved_count += 1
                    
                except Exception as e:
                    st.warning(f"항목 저장 중 오류: {str(e)}")
            
            return response_data.get('items', []), response_data.get('total', 0), saved_count
        else:
            st.error(f"네이버 API 오류: {response_code}")
            return [], 0, 0
            
    except Exception as e:
        st.error(f"네이버 검색 중 오류 발생: {str(e)}")
        return [], 0, 0

def semantic_search(query_text, source_type="블로그", limit=10, match_threshold=0.5):
    """시맨틱 검색 수행"""
    try:
        # 쿼리 텍스트에 대한 임베딩 생성
        query_embedding = generate_embedding(query_text)
        
        # match_documents 함수를 사용한 벡터 검색
        try:
            response = supabase.rpc(
                'match_documents', 
                {
                    'query_embedding': query_embedding,
                    'match_threshold': match_threshold,
                    'match_count': limit * 2  # 필터링 후 충분한 결과를 위해 더 많이 가져옴
                }
            ).execute()
            
            # 전체 검색 결과 수 표시 (디버깅용)
            st.sidebar.info(f"검색 결과 총 {len(response.data)}개")
            
            if response.data and len(response.data) > 0:
                # 클라이언트 측에서 소스 타입에 따라 필터링
                filtered_results = []
                for item in response.data:
                    # metadata 확인
                    metadata = item.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    item_source_type = metadata.get('collection', '')
                    
                    # 소스 타입이 일치하는 경우에만 추가 (대소문자 무시 및 부분 일치 검색)
                    if source_type.lower() in item_source_type.lower():
                        filtered_results.append(item)
                
                # 필터링 후 결과 수 표시 (디버깅용)
                st.sidebar.info(f"{source_type} 필터링 후 {len(filtered_results)}개")
                
                # 최대 limit 개수만큼 결과 반환
                return filtered_results[:limit]
            else:
                return []
                
        except Exception as e:
            st.sidebar.warning(f"시맨틱 검색 실패: {str(e)}")
            return []
        
    except Exception as e:
        st.error(f"시맨틱 검색 중 오류 발생: {str(e)}")
        raise

def get_system_prompt(source_type):
    """소스 타입에 따른 시스템 프롬프트 생성"""
    if source_type == "블로그":
        return """당신은 네이버 블로그 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다.
블로그 글은 개인의 경험과 의견을 담고 있으므로, 주관적인 내용이 포함될 수 있음을 인지하세요.
여러 블로그의 정보를 종합하여 균형 잡힌 시각을 제공하되, 정보의 출처가 개인 블로그임을 명시하세요.
특히 레시피, DIY 방법, 여행 경험 등 실용적인 정보에 집중하되, 의학적 조언이나 전문적인 내용은 참고 정보로만 안내하세요."""

    elif source_type == "뉴스":
        return """당신은 네이버 뉴스 데이터를 기반으로 정확하고 객관적인 정보를 제공하는 도우미입니다.
뉴스 기사의 사실과 정보를 전달할 때는 편향되지 않게 중립적인 입장을 유지하세요.
여러 언론사의 기사를 비교하여 다양한 관점을 제시하고, 정보의 출처와 발행 날짜를 명확히 하세요.
특히 시사 문제, 최신 이슈, 사회 현상에 대해 설명할 때는 다양한 의견이 있을 수 있음을 인지하세요."""

    elif source_type == "쇼핑":
        return """당신은 네이버 쇼핑 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다.
상품 정보, 가격, 기능, 특징 등을 객관적으로 설명하고 비교하세요.
다양한 상품 옵션과 가격대를 안내하되, 특정 브랜드나 제품을 지나치게 홍보하지 마세요.
사용자의 요구에 맞는 상품 추천이나 구매 팁을 제공할 때는 실용적인 관점에서 접근하세요."""

    else:
        return """당신은 네이버 검색 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다.
주어진 문서들의 내용만 사용하여 사용자 질문에 맞는 최적의 답변을 제공하세요.
문서에 없는 내용은 추가하지 말고 정확한 사실만 전달하세요."""

def get_user_prompt(query, context_text, source_type):
    """소스 타입에 따른 사용자 프롬프트 생성"""
    if source_type == "블로그":
        return f"""다음은 네이버 블로그에서 수집한 데이터입니다:

{context_text}

위 블로그 글들을 바탕으로 다음 질문에 상세히 답변해주세요: 
"{query}"

답변 작성 규칙:
1. 한국어로 자연스럽게 답변해주세요.
2. 블로그 글은 개인의 경험과 의견을 담고 있으므로, 정보의 주관성을 고려해주세요.
3. 여러 블로그의 공통된 내용에 중점을 두고, 개인적 경험이나 팁은 "블로거의 경험에 따르면..."과 같이 맥락을 제공해주세요.
4. 블로그 글들 간에 상충되는 정보가 있다면 "일부 블로거는 A를 추천하는 반면, 다른 블로거는 B를 선호합니다"와 같이 다양한 의견을 제시해주세요.
5. 레시피, DIY 방법, 여행 경험 등 실용적인 정보에 집중해주세요.
6. 출처를 명시할 때는 "문서 2의 블로거에 따르면..."과 같이 표현해주세요.
7. 제공된 문서 내용만 사용하고, 문서에 없는 내용은 추측하거나 답변하지 마세요."""

    elif source_type == "뉴스":
        return f"""다음은 네이버 뉴스에서 수집한, 신뢰할 수 있는 언론사의 기사입니다:

{context_text}

위 뉴스 기사들을 바탕으로 다음 질문에 상세히 답변해주세요: 
"{query}"

답변 작성 규칙:
1. 한국어로 자연스럽게 답변해주세요.
2. 뉴스 기사의 사실과 정보를 전달할 때는 편향되지 않게 중립적인 입장을 유지하세요.
3. 기사의 발행 날짜를 고려하여 정보의 시의성을 명시하세요. (예: "2023년 5월 보도에 따르면...")
4. 여러 언론사의 기사를 인용할 때는 "문서 1의 OO일보에 따르면..."와 같이 출처를 명확히 하세요.
5. 기사들 간에 상충되는 정보가 있다면 이를 언급하고 각 관점을 공정하게 제시하세요.
6. 제공된 기사 내용만 사용하고, 기사에 없는 내용은 추측하거나 답변하지 마세요."""

    elif source_type == "쇼핑":
        return f"""다음은 네이버 쇼핑에서 수집한 상품 정보입니다:

{context_text}

위 쇼핑 데이터를 바탕으로 다음 질문에 상세히 답변해주세요: 
"{query}"

답변 작성 규칙:
1. 한국어로 자연스럽게 답변해주세요.
2. 상품의 가격, 기능, 특징 등을 객관적으로 설명하고 비교해주세요.
3. 가격은 범위로 표현하고 정확한 가격이 있다면 언급해주세요. (예: "이 제품은 30,000원에서 50,000원 사이의 가격대를 형성하고 있습니다")
4. 다양한 브랜드와 제품을 균형 있게 소개하고, 특정 상품을 지나치게 홍보하지 마세요.
5. 상품의 특징을 비교할 때는 "A 제품은 X 기능이 있지만, B 제품은 Y 기능이 강조됩니다"와 같이 객관적으로 설명해주세요.
6. 제공된 상품 정보만 사용하고, 문서에 없는 내용은 추측하거나 답변하지 마세요."""

    else:
        return f"""다음은 네이버 검색에서 수집한 데이터입니다:

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
7. 필요한 경우 정보의 출처를 언급해주세요(예: "문서 2에 따르면...")."""

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
            
            # metadata 확인 (JSON 문자열일 경우 파싱)
            metadata = result.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
                    
            title = metadata.get('title', '제목 없음')
            date = metadata.get('date', '')  # 날짜 정보가 있으면 추가
            
            # 날짜 정보가 있으면 포함
            date_info = f" (작성일: {date})" if date else ""
            
            # 소스 타입에 맞는 추가 정보
            if source_type == "블로그" and 'bloggername' in metadata:
                source_info = f" - 블로거: {metadata['bloggername']}"
            elif source_type == "뉴스" and 'publisher' in metadata:
                source_info = f" - 출처: {metadata['publisher']}"
            elif source_type == "쇼핑" and 'mallname' in metadata:
                price_info = f", 가격: {metadata.get('lprice', '정보 없음')}원" if 'lprice' in metadata else ""
                source_info = f" - 판매처: {metadata['mallname']}{price_info}"
            else:
                source_info = ""
            
            # 출처 타입과 함께 컨텍스트 추가
            contexts.append(f"문서 {i+1} - [{source_type}] {title}{date_info}{source_info}:\n{content}\n")
        
        context_text = "\n".join(contexts)
        
        # 소스 타입에 맞는 프롬프트 생성
        system_prompt = get_system_prompt(source_type)
        user_prompt = get_user_prompt(query, context_text, source_type)

        # GPT-4o-mini로 답변 생성
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
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
st.write("시맨틱 검색 기술을 이용하여 네이버 데이터를 검색하고 질문에 답변합니다.")

# 검색 모드 선택
search_mode = st.sidebar.radio(
    "검색 모드 선택", 
    options=["시맨틱 검색 (저장된 데이터)", "새 데이터 수집 및 저장"], 
    index=0
)

# 검색 소스 선택 (라디오 버튼) - 가로로 배치
source_type = st.radio(
    "검색 소스 선택", 
    options=["블로그", "뉴스", "쇼핑"], 
    index=0,
    horizontal=True  # 가로로 배치
)

# 검색 입력 - 소스 타입에 따라 다른 예시 질문 제공
if source_type == "블로그":
    default_query = "안성탕면 맛있게 끓이는 방법이 뭐지?"
    help_text = "블로그 데이터에서 레시피, 리뷰, 여행 경험 등을 검색해보세요"
elif source_type == "뉴스":
    default_query = "최근 경제 이슈는 무엇인가요?"
    help_text = "뉴스 데이터에서 시사, 경제, 사회 이슈 등을 검색해보세요"
elif source_type == "쇼핑":
    default_query = "인기있는 노트북은 어떤 것이 있나요?"
    help_text = "쇼핑 데이터에서 상품 정보, 가격 비교, 구매 팁 등을 검색해보세요"

# 검색 입력
query = st.text_input("질문 입력", value=default_query, help=help_text)

# 원본 검색 결과 표시 옵션
show_raw_results = st.sidebar.checkbox("원본 검색 결과 표시", value=True)

# 검색 결과 수 및 유사도 설정
if search_mode == "시맨틱 검색 (저장된 데이터)":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        result_count = st.slider("검색 결과 수", min_value=3, max_value=20, value=10)
    with col2:
        similarity_threshold = st.slider("유사도 임계값", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
else:
    result_count = st.sidebar.slider("검색 결과 수", min_value=5, max_value=50, value=20)

# 검색 버튼
search_button_text = "시맨틱 검색" if search_mode == "시맨틱 검색 (저장된 데이터)" else "데이터 수집 및 저장"
if st.button(f"{source_type} {search_button_text}", key="search_button"):
    if query:
        if search_mode == "시맨틱 검색 (저장된 데이터)":
            # 시맨틱 검색 모드
            with st.spinner(f"{source_type} 시맨틱 검색 중..."):
                try:
                    # 시맨틱 검색 수행
                    results = semantic_search(query, source_type=source_type, limit=result_count, match_threshold=similarity_threshold)
                    
                    if results:
                        st.success(f"{len(results)}개의 {source_type} 결과를 찾았습니다.")
                        
                        # GPT로 답변 생성
                        with st.spinner("AI 에이전트 답변 생성 중..."):
                            gpt_answer = generate_answer_with_gpt(query, results, source_type)
                            
                            # 답변 표시
                            st.markdown(f"## AI 답변 ({source_type} 데이터 기반)")
                            st.markdown(gpt_answer)
                            
                            # 구분선
                            st.markdown("---")
                        
                        # 시각화 생성 및 표시
                        visualizations = visualize_search_results(results, source_type)
                        if visualizations:
                            st.markdown("## 검색 결과 분석")
                            
                            # 유사도 그래프 표시
                            if 'similarity' in visualizations:
                                st.plotly_chart(visualizations['similarity'], use_container_width=True)
                            
                            # 키워드 빈도 분석 표시
                            vis_col1, vis_col2 = st.columns(2)
                            
                            with vis_col1:
                                if 'wordcloud' in visualizations:
                                    st.markdown("### 주요 키워드 워드클라우드")
                                    st.pyplot(visualizations['wordcloud'])
                                elif 'word_freq' in visualizations:
                                    st.markdown("### 주요 키워드 빈도")
                                    st.plotly_chart(visualizations['word_freq'], use_container_width=True)
                            
                            with vis_col2:
                                # 소스 타입별 추가 시각화
                                if source_type == "블로그" and 'blogger_dist' in visualizations:
                                    st.markdown("### 블로거별 분포")
                                    st.plotly_chart(visualizations['blogger_dist'], use_container_width=True)
                                elif source_type == "뉴스" and 'publisher_dist' in visualizations:
                                    st.markdown("### 언론사별 분포")
                                    st.plotly_chart(visualizations['publisher_dist'], use_container_width=True)
                                elif source_type == "쇼핑" and 'price_dist' in visualizations:
                                    st.markdown("### 가격대별 분포")
                                    st.plotly_chart(visualizations['price_dist'], use_container_width=True)
                        
                        # 원본 검색 결과 표시 옵션
                        if show_raw_results:
                            st.markdown(f"## {source_type} 검색 결과 원본")
                            for i, result in enumerate(results):
                                similarity = result['similarity'] * 100  # 백분율로 변환
                                
                                # 메타데이터 확인 (JSON 문자열일 경우 파싱)
                                metadata = result.get('metadata', {})
                                if isinstance(metadata, str):
                                    try:
                                        metadata = json.loads(metadata)
                                    except:
                                        metadata = {}
                                
                                title = metadata.get('title', '제목 없음')
                                
                                # URL 추출
                                url = metadata.get('url', None)
                                
                                # 결과 표시
                                with st.expander(f"{i+1}. {title} (유사도: {similarity:.2f}%)"):
                                    st.write(f"**내용:** {result['content']}")
                                    
                                    # 메타데이터 정보 표시
                                    meta_col1, meta_col2 = st.columns(2)
                                    
                                    with meta_col1:
                                        if source_type == "블로그" and 'bloggername' in metadata:
                                            st.write(f"**블로거:** {metadata['bloggername']}")
                                        elif source_type == "뉴스" and 'publisher' in metadata:
                                            st.write(f"**언론사:** {metadata['publisher']}")
                                        elif source_type == "쇼핑" and 'maker' in metadata:
                                            st.write(f"**제조사:** {metadata['maker']}")
                                            
                                        if 'date' in metadata:
                                            st.write(f"**날짜:** {metadata['date']}")
                                    
                                    with meta_col2:
                                        if url:
                                            st.markdown(f"**링크:** [원본 보기]({url})")
                                        if source_type == "쇼핑":
                                            if 'lprice' in metadata:
                                                st.write(f"**최저가:** {metadata['lprice']}원")
                                            if 'mallname' in metadata:
                                                st.write(f"**판매처:** {metadata['mallname']}")
                    else:
                        st.warning(f"{source_type}에서 검색 결과가 없습니다. 새 데이터를 수집하거나 다른 검색어를 시도해보세요.")
                
                except Exception as e:
                    st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
        
        else:
            # 네이버 API 검색 및 저장 모드
            with st.spinner(f"네이버 {source_type} API 검색 및 데이터 저장 중..."):
                try:
                    # 네이버 API 검색 수행 및 Supabase에 저장
                    items, total_count, saved_count = search_naver_api(query, source_type, result_count)
                    
                    if items:
                        st.success(f"네이버 {source_type}에서 총 {total_count}개 중 {len(items)}개의 결과를 찾았고, {saved_count}개를 새로 저장했습니다.")
                        
                        # 저장 후 즉시 시맨틱 검색 수행
                        with st.spinner("저장된 데이터로 시맨틱 검색 중..."):
                            # 잠시 대기 (데이터베이스 저장 완료 대기)
                            import time
                            time.sleep(2)
                            
                            # 시맨틱 검색 수행
                            results = semantic_search(query, source_type=source_type, limit=result_count, match_threshold=0.5)
                            
                            if results:
                                # GPT로 답변 생성
                                with st.spinner("AI 에이전트 답변 생성 중..."):
                                    gpt_answer = generate_answer_with_gpt(query, results, source_type)
                                    
                                    # 답변 표시
                                    st.markdown(f"## AI 답변 ({source_type} 데이터 기반)")
                                    st.markdown(gpt_answer)
                                    
                                    # 구분선
                                    st.markdown("---")
                                
                                # 시각화 생성 및 표시
                                visualizations = visualize_search_results(results, source_type)
                                if visualizations:
                                    st.markdown("## 검색 결과 분석")
                                    
                                    # 유사도 그래프 표시
                                    if 'similarity' in visualizations:
                                        st.plotly_chart(visualizations['similarity'], use_container_width=True)
                                    
                                    # 키워드 빈도 분석 표시
                                    vis_col1, vis_col2 = st.columns(2)
                                    
                                    with vis_col1:
                                        if 'wordcloud' in visualizations:
                                            st.markdown("### 주요 키워드 워드클라우드")
                                            st.pyplot(visualizations['wordcloud'])
                                        elif 'word_freq' in visualizations:
                                            st.markdown("### 주요 키워드 빈도")
                                            st.plotly_chart(visualizations['word_freq'], use_container_width=True)
                                    
                                    with vis_col2:
                                        # 소스 타입별 추가 시각화
                                        if source_type == "블로그" and 'blogger_dist' in visualizations:
                                            st.markdown("### 블로거별 분포")
                                            st.plotly_chart(visualizations['blogger_dist'], use_container_width=True)
                                        elif source_type == "뉴스" and 'publisher_dist' in visualizations:
                                            st.markdown("### 언론사별 분포")
                                            st.plotly_chart(visualizations['publisher_dist'], use_container_width=True)
                                        elif source_type == "쇼핑" and 'price_dist' in visualizations:
                                            st.markdown("### 가격대별 분포")
                                            st.plotly_chart(visualizations['price_dist'], use_container_width=True)
                            else:
                                st.warning("데이터는 저장되었지만 시맨틱 검색에서 관련 결과를 찾지 못했습니다. 잠시 후 다시 시도해 보세요.")
                        
                        # 네이버 API 결과 표시
                        if show_raw_results:
                            st.markdown(f"## 네이버 {source_type} 검색 결과")
                            
                            # 데이터프레임으로 표시할 데이터 준비
                            df_data = []
                            for i, item in enumerate(items):
                                # HTML 태그 제거
                                title = re.sub('<[^<]+?>', '', item.get('title', ''))
                                
                                # 소스 타입별 표시 항목
                                if source_type == "블로그":
                                    description = re.sub('<[^<]+?>', '', item.get('description', ''))
                                    df_data.append({
                                        '제목': title,
                                        '내용 미리보기': description[:100] + "...",
                                        '블로거': item.get('bloggername', ''),
                                        '날짜': item.get('postdate', ''),
                                        '링크': item.get('link', '')
                                    })
                                elif source_type == "뉴스":
                                    description = re.sub('<[^<]+?>', '', item.get('description', ''))
                                    df_data.append({
                                        '제목': title,
                                        '내용 미리보기': description[:100] + "...",
                                        '언론사': item.get('publisher', ''),
                                        '날짜': item.get('pubDate', ''),
                                        '링크': item.get('link', '')
                                    })
                                elif source_type == "쇼핑":
                                    df_data.append({
                                        '제품명': title,
                                        '가격': f"{item.get('lprice', '')}원",
                                        '판매처': item.get('mallName', ''),
                                        '제조사': item.get('maker', ''),
                                        '링크': item.get('link', '')
                                    })
                            
                            # 데이터프레임 생성 및 표시
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # 각 결과 상세 내용 표시
                            for i, item in enumerate(items):
                                title = re.sub('<[^<]+?>', '', item.get('title', ''))
                                
                                with st.expander(f"{i+1}. {title}"):
                                    if source_type in ["블로그", "뉴스"]:
                                        description = re.sub('<[^<]+?>', '', item.get('description', ''))
                                        st.write(f"**내용:** {description}")
                                    
                                    # 메타데이터 정보 표시
                                    meta_col1, meta_col2 = st.columns(2)
                                    
                                    with meta_col1:
                                        if source_type == "블로그":
                                            st.write(f"**블로거:** {item.get('bloggername', '')}")
                                            st.write(f"**날짜:** {item.get('postdate', '')}")
                                        elif source_type == "뉴스":
                                            st.write(f"**언론사:** {item.get('publisher', '')}")
                                            st.write(f"**날짜:** {item.get('pubDate', '')}")
                                        elif source_type == "쇼핑":
                                            st.write(f"**제조사:** {item.get('maker', '')}")
                                            st.write(f"**브랜드:** {item.get('brand', '')}")
                                    
                                    with meta_col2:
                                        st.markdown(f"**링크:** [원본 보기]({item.get('link', '')})")
                                        if source_type == "쇼핑":
                                            st.write(f"**최저가:** {item.get('lprice', '')}원")
                                            st.write(f"**판매처:** {item.get('mallName', '')}")
                    else:
                        st.warning(f"네이버 {source_type}에서 검색 결과가 없습니다. 다른 검색어나 다른 소스 타입으로 시도해보세요.")
                
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
    st.sidebar.info(f"저장된 총 문서 수: {doc_count}개")
    
    # 각 소스 타입별 문서 수 표시 시도
    try:
        collections = {}
        collection_query = supabase.table('documents').select('metadata').execute()
        for item in collection_query.data:
            metadata = item.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    continue
            
            collection = metadata.get('collection', '기타')
            if collection in collections:
                collections[collection] += 1
            else:
                collections[collection] = 1
        
        # 소스 타입별 문서 수 표시
        for collection, count in collections.items():
            st.sidebar.info(f"{collection} 문서 수: {count}개")
    except:
        pass
except Exception as e:
    st.sidebar.error("데이터베이스 상태를 확인할 수 없습니다.")

# 사용 안내
st.sidebar.title("사용 안내")
st.sidebar.info(f"""
**검색 모드:**
1. **시맨틱 검색 (저장된 데이터)**: 이미 저장된 데이터를 의미 기반으로 검색합니다.
2. **새 데이터 수집 및 저장**: 네이버 API에서 새 데이터를 가져와 저장하고 검색합니다.

**검색 소스 선택:** 블로그, 뉴스, 쇼핑 중에서 검색할 소스를 선택하세요.

**유사도 임계값:** 시맨틱 검색에서 얼마나 유사한 결과를 포함할지 결정합니다. 
- 높음 (0.8~1.0): 매우 관련성 높은 결과만 표시
- 중간 (0.5~0.7): 균형잡힌 관련성 (권장)
- 낮음 (0.1~0.4): 더 많은 결과를 포함하지만 관련성이 낮을 수 있음

💡 팁: 각 소스 타입에 적합한 질문을 입력하세요:
- 블로그: 레시피, 여행 경험, 리뷰, DIY 방법 등
- 뉴스: 시사 이슈, 사회 현상, 경제 동향 등
- 쇼핑: 상품 정보, 가격 비교, 구매 팁 등
""")

# 네이버 API 정보
st.sidebar.title("네이버 API 정보")
st.sidebar.info("""
- Client ID: 9XhhxLV1IzDpTZagoBr1
- 데이터 출처: 네이버 검색 API
""")
