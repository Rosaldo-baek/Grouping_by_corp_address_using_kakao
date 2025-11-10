import streamlit as st
import pandas as pd
import time
import math
import requests
import os

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
from openai import OpenAI

GPT_ADDRESS_SYSTEM_PROMPT = """
당신은 주소 정제 전문가입니다. 

주어진 사업체 주소를 주소 어플에서 위치를 검색가능하게 불필요한 부분은 없애고, 정제하는 작업을 수행합니다. 
일반적인 주소는 도로명 형태 - 시/도 + 시/군/구 + (읍/면/동) + 도로명 + 건물번호 이거나 
지번 - 시/도 + 시/군/구 + (읍/면/동/리) + 지번 이야 

아래 규칙에 따라 주소를 정제하세요

1) 지번번호가 여러개 들어가 있는 경우 맨 처음 하나만 남기세요 
예) 정제전: 경기 남양주시 오남읍 양지로281번길 78-19 78-23  정제후: 경기 남양주시 오남읍 양지로281번길 78-19 

2) 세부주소 (예 - 호수)가 들어가면 해당 호수 번호는 지웁니다. 
예) 정제전: 경남 김해시 금관대로1134번길 16 103-2502 정제후: 경남 김해시 금관대로1134번길 16 
예) 정제전: 경기 안양시 만안구 소곡로26번길 12 302호 정제후: 경기 안양시 만안구 소곡로26번길 12

3) 주소 끝에 있는 가로 안 형태로 ( )불필요한 건물명, 동/리등 이 있는 경우 삭제 
예) 정제전: 경남 김해시 금관대로 1134번길 16 (외동, 쌍용 더 플래티넘 김해) 정제후: 경남 김해시 금관대로 1134번길 16
예) 정제전: 전남 장성군 삼서면 보생로 109 (보생리) 정제후: 전남 장성군 삼서면 보생로 109

정제할 요소가 없으면 원 주소를 그대로 반환합니다. 
임의로 주소 내 숫자 및 기초 시도, 시군구 정보를 수정하는 것은 금지합니다. 

반드시 주어진 주소에 대한 정제 주소만 응답하십시오.
앞에 번호는 붙일 필요가 없습니다.
예) 
1. 주소 
2. 주소 
이렇게 숫자 붙이지 마세요.

"""


def format_radius_suffix(radius_km: float) -> str:
    """
    5.0 -> '5km', 2.5 -> '2.5km' 형태 suffix 생성
    """
    s = f"{radius_km:.3f}".rstrip('0').rstrip('.')
    return f"{s}km"

def clean_addresses_with_gpt_batch(
    df: pd.DataFrame,
    address_col: str,
    client: OpenAI,
    batch_size: int = 100,
    progress_hook=None,
    output_col: str = "정제주소",
) -> pd.DataFrame:
    addrs = df[address_col].astype(str).tolist()
    total = len(addrs)
    cleaned = [None] * total

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = addrs[start:end]

        # 번호 매겨서 전달
        user_content = "아래 주소들을 번호에 맞게 한 줄에 하나씩 정제해서 반환하세요.\n"
        user_content += "\n".join(f"{i+1}. {a}" for i, a in enumerate(batch))

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": GPT_ADDRESS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=256 * batch_size // 50,  # 대략 줄수에 맞게
        )
        text = resp.choices[0].message.content.strip()

        # GPT에게 "1. ~ 2. ~" 또는 줄 단위로만 보내달라고 강하게 명시해야 함
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # 길이 안 맞을 수 있으므로 방어적으로 매핑
        for i in range(start, end):
            local_idx = i - start
            if local_idx < len(lines):
                cleaned[i] = lines[local_idx]
            else:
                cleaned[i] = addrs[i]  # 실패 시 원본 유지

        if progress_hook is not None:
            progress_hook(end, total)

    out = df.copy()
    out[output_col] = cleaned
    return out
# ----------------------------------------------------
# 2) Kakao API 세션 및 검색 함수들
#    (API Key는 Streamlit UI에서 입력받아 헤더에 설정)
# ----------------------------------------------------
BASE_URL = "https://dapi.kakao.com/v2/local/search/address.json"
session = requests.Session()  # 헤더는 main flow에서 업데이트

def search_address_once(query: str, page: int = 1, size: int = 10, timeout: float = 5.0):
    if not query or not isinstance(query, str):
        return None

    params = {"query": query, "page": page, "size": size}
    try:
        resp = session.get(BASE_URL, params=params, timeout=timeout)
        # 429/5xx 대비
        if resp.status_code == 429:
            # 속도 제한. 호출자 레벨에서 재시도 권장
            return {"_error": "rate_limited", "_status": 429}
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"_error": str(e), "_status": getattr(e.response, "status_code", None)}


def parse_first_document(payload: dict):
    """
    카카오 응답에서 첫 문서만 사용.
    """
    if not payload or "_error" in payload:
        return {
            "kakao_x": None, "kakao_y": None,
            "kakao_address_name": None, "kakao_address_type": None,
            "kakao_meta_total_count": None,
            "kakao_error": payload.get("_error") if isinstance(payload, dict) else None,
            "kakao_status": payload.get("_status") if isinstance(payload, dict) else None,
        }

    docs = payload.get("documents", [])
    meta = payload.get("meta", {}) or {}
    if not docs:
        return {
            "kakao_x": None, "kakao_y": None,
            "kakao_address_name": None, "kakao_address_type": None,
            "kakao_meta_total_count": meta.get("total_count"),
            "kakao_error": None, "kakao_status": 200,
        }

    d0 = docs[0]
    return {
        "kakao_x": d0.get("x"),
        "kakao_y": d0.get("y"),
        "kakao_address_name": d0.get("address_name"),
        "kakao_address_type": d0.get("address_type"),
        "kakao_meta_total_count": meta.get("total_count"),
        "kakao_error": None, "kakao_status": 200,
    }


def query_with_backoff(addr: str, max_retries: int = 3, base_sleep: float = 0.5):
    """429 또는 일시 오류 시 지수형 백오프 적용."""
    attempt = 0
    while True:
        payload = search_address_once(addr)
        # 정상 또는 영구 오류면 종료
        if not payload or "_error" not in payload or payload.get("_status") not in (429, 500, 502, 503, 504):
            return parse_first_document(payload)
        # 재시도
        if attempt >= max_retries:
            return parse_first_document(payload)
        sleep_s = base_sleep * (2 ** attempt)
        time.sleep(sleep_s)
        attempt += 1


def enrich_df_with_kakao(
    df: pd.DataFrame,
    address_col: str = "주소",
    parallel_chunks: int = 1,
    progress_hook=None,):
    """
    기본은 순차 처리. 대량일 경우 parallel_chunks>1로 나눠 처리 가능(멀티프로세싱 직접 추가하면 됨).
    progress_hook: 각 건 처리 후 진행률 업데이트용 콜백
                   시그니처: progress_hook(done: int, total: int)
    """
    if address_col not in df.columns:
        raise ValueError(f"'{address_col}' 컬럼 없음")

    total = len(df)
    records = []

    for i, addr in enumerate(df[address_col]):
        rec = query_with_backoff(addr)
        records.append(rec)

        if progress_hook is not None:
            # i는 0부터 시작이므로 +1
            progress_hook(i + 1, total)

    res_df = pd.DataFrame(records)

    # 원본과 병합
    out = pd.concat([df.reset_index(drop=True), res_df], axis=1)
    return out

# ----------------------------------------------------
# 3) 반경 기반 그룹핑 함수 (suffix 자동 적용)
# ----------------------------------------------------
def group_by_radius_complete_link(df_in: pd.DataFrame,
                                  lat_col: str = "kakao_y",
                                  lon_col: str = "kakao_x",
                                  radius_km: float = 5.0):
    """
    radius_km에 따라 컬럼명 suffix 자동 설정
    예: radius_km=5.0 -> suffix='5km'
    """
    suffix = format_radius_suffix(radius_km)

    g = df_in.dropna(subset=[lat_col, lon_col]).copy()
    idx_map = g.index.to_numpy()

    coords_rad = np.radians(g[[lat_col, lon_col]].to_numpy())
    D_rad = haversine_distances(coords_rad)  # 라디안
    D_km = D_rad * 6371.0

    # complete-linkage로 직경 제약
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=radius_km,
        metric='precomputed',
        linkage='complete',
        compute_full_tree=True
    )
    labels = model.fit_predict(D_km)

    comp_sizes = pd.Series(labels).value_counts().to_dict()
    size_arr = np.vectorize(comp_sizes.get)(labels)
    isolated = (size_arr == 1)

    component_col = f"component_id_{suffix}"
    size_col = f"component_size_{suffix}"
    iso_col = f"is_isolated_{suffix}"
    group_col = f"group_key_{suffix}"

    group_keys = np.where(
        isolated,
        np.array([f"I{ix}" for ix in idx_map]),
        np.array([f"C{c}" for c in labels])
    )

    out = df_in.copy()
    out.loc[idx_map, component_col] = labels
    out.loc[idx_map, size_col] = size_arr
    out.loc[idx_map, iso_col] = isolated
    out.loc[idx_map, group_col] = group_keys

    summary = (
        out.loc[idx_map, [group_col, component_col, size_col, lat_col, lon_col]]
        .assign(_one=1)
        .groupby([group_col, component_col, size_col], as_index=False)
        .agg(
            n_points=("_one", "sum"),
            center_lat=(lat_col, "mean"),
            center_lon=(lon_col, "mean"),
        )
        .sort_values(["n_points", group_col], ascending=[False, True])
    )

    # 사용한 컬럼명을 함께 반환해 후속 로직에서 사용
    col_info = {
        "component_col": component_col,
        "size_col": size_col,
        "iso_col": iso_col,
        "group_col": group_col,
        "suffix": suffix,
    }
    return out, summary, col_info


st.title("카카오 주소 → 좌표 추출 및 반경 기반 그룹핑")

st.markdown("#### 1. API 키 입력")
gpt_api_key = st.text_input("OpenAI GPT API 키를 입력하세요", type="password")
kakao_api_key = st.text_input("Kakao REST API 키를 입력하세요", type="password")
st.caption("※ Kakao 로컬 API는 계정 기준 월 최대 3,000,000건 조회 제한이 있으니 호출량에 유의해 주세요.\n 사용량은 kakao developers 내 쿼터 참고")

st.markdown("#### 2. 분석 옵션 설정")
radius_km = st.slider("클러스터링 반경 (km)", min_value=1.0, max_value=20.0, step=0.5, value=5.0)
min_group_size = st.number_input("유효 집단 최소 건수 (기본 20)", min_value=1, value=20, step=1)

st.markdown("#### 3. 엑셀 파일 업로드")
uploaded_file = st.file_uploader("그루핑 대상 엑셀 파일을 업로드하세요", type=["xlsx"])

run_button = st.button("주소 좌표 추출 및 그룹핑 실행")

if run_button:
    if not gpt_api_key:
        st.error("OpenAI GPT API 키를 입력해야 합니다.")
        st.stop()
    if not kakao_api_key:
        st.error("Kakao REST API 키를 입력해야 합니다.")
        st.stop()
    if uploaded_file is None:
        st.error("엑셀 파일을 업로드해야 합니다.")
        st.stop()

    # 클라이언트 및 세션 설정
    gpt_client = OpenAI(api_key=gpt_api_key)
    session.headers.update({"Authorization": f"KakaoAK {kakao_api_key}"})

    # 엑셀 로드
    try:
        df_origin = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류 발생: {e}")
        st.stop()

    # 원본에서 마지막 2개 컬럼 제외 (기존 로직 유지)
    col_list = df_origin.columns[:-2]
    df = df_origin[col_list].copy()

    st.write("원본 데이터 preview:")
    st.dataframe(df.head())
    st.markdown("#### 4. 주소 정제 실행 (GPT 사용)")

    gpt_progress = st.progress(0)
    gpt_status = st.empty()

    def gpt_progress_hook(done: int, total: int):
        if total <= 0:
            return
        ratio = done / total
        gpt_progress.progress(ratio)
        gpt_status.text(f"주소 정제 진행률: {done}/{total} ({ratio*100:.1f}%)")

    with st.spinner("GPT를 이용해 주소 정제 중입니다."):
        df_clean = clean_addresses_with_gpt_batch(
            df,
            address_col="주소",      # 원본 주소 컬럼명
            client=gpt_client,
            progress_hook=gpt_progress_hook,
            output_col="정제주소",
        )

    gpt_progress.empty()
    gpt_status.empty()

    st.success("주소 정제 완료")
    st.write("정제된 주소 preview:")
    st.dataframe(df_clean[["주소", "정제주소"]].head())
    # ------------------------------------------------

    # ------------------------------------------------
    st.markdown("#### 5. 카카오 주소 검색 실행")
    
    kakao_progress_bar = st.progress(0)
    kakao_status_text = st.empty()

    def kakao_progress_hook(done: int, total: int):
        if total <= 0:
            return
        ratio = done / total
        kakao_progress_bar.progress(ratio)
        kakao_status_text.text(
            f"카카오 주소 조회 진행률: {done}/{total} ({ratio*100:.1f}%)"
        )

    with st.spinner("카카오 주소 검색 및 좌표 추출 중입니다."):
        out = enrich_df_with_kakao(
            df_clean,
            address_col="정제주소",
            progress_hook=kakao_progress_hook,
        )

    kakao_progress_bar.empty()
    kakao_status_text.empty()

    st.success("카카오 좌표 추출 완료")
    st.write("좌표가 포함된 데이터 preview:")
    st.dataframe(out.head())

    # 숫자형 변환
    df_valid = out.copy()
    df_valid["kakao_x"] = pd.to_numeric(df_valid["kakao_x"], errors="coerce")
    df_valid["kakao_y"] = pd.to_numeric(df_valid["kakao_y"], errors="coerce")

    # ------------------------------------------------
    # 4-2) 반경 기반 그룹핑
    # ------------------------------------------------
    st.markdown("#### 5. 반경 기반 그룹핑 실행")
    with st.spinner("거리 기반 클러스터링 중..."):
        labeled_df, summary, col_info = group_by_radius_complete_link(
            df_valid,
            lat_col="kakao_y",
            lon_col="kakao_x",
            radius_km=radius_km,
        )

    suffix = col_info["suffix"]
    group_col = col_info["group_col"]

    st.success(f"그룹핑 완료 (반경 {suffix} 기준)")

    st.write("그룹 요약 정보 일부:")
    st.dataframe(summary.head())

    # ------------------------------------------------
    # 4-3) 최소 건수 이상 집단 필터링
    # ------------------------------------------------
    st.markdown("#### 6. 최소 건수 이상 집단 필터링")
    vc = labeled_df[group_col].value_counts()
    valid_keys = vc[vc >= min_group_size].index

    labeled_df_sec = labeled_df.loc[labeled_df[group_col].isin(valid_keys)].copy()
    n_groups = labeled_df_sec[group_col].nunique()

    st.write(f"최소 {min_group_size}건 이상 집단 수: {n_groups}개")
    st.dataframe(labeled_df_sec.head())

    # ------------------------------------------------
    # 4-4) 파일 저장 (로컬 + 다운로드 버튼)
    # ------------------------------------------------
    # 로컬 저장 (원래 파일명 패턴 유지하되 suffix 반영)
    base_suffix = suffix  # 예: '5km'
    full_path_all = f"그룹핑 테스트_최종완료_{base_suffix}_인접_집단분류.csv"
    full_path_valid = f"그룹핑 테스트_최종완료_{base_suffix}_인접_{min_group_size}개 이상 집단.csv"

    try:
        labeled_df.to_csv(full_path_all, encoding="euc-kr", index=False)
        labeled_df_sec.to_csv(full_path_valid, encoding="euc-kr", index=False)
        st.info(f"로컬에 다음 파일이 저장됨:\n- {full_path_all}\n- {full_path_valid}")
    except Exception as e:
        st.warning(f"로컬 파일 저장 중 오류가 발생했지만, 아래에서 직접 다운로드는 가능합니다. 오류: {e}")

    # 다운로드 버튼용 버퍼 생성 (UTF-8 + BOM 또는 euc-kr 중 선택 가능)
    all_csv_bytes = labeled_df.to_csv(index=False, encoding="euc-kr").encode("euc-kr", errors="ignore")
    valid_csv_bytes = labeled_df_sec.to_csv(index=False, encoding="euc-kr").encode("euc-kr", errors="ignore")

    st.download_button(
        label=f"전체 결과 다운로드 ({base_suffix})",
        data=all_csv_bytes,
        file_name=full_path_all,
        mime="text/csv",
    )

    st.download_button(
        label=f"최소 {min_group_size}건 이상 집단 결과 다운로드 ({base_suffix})",
        data=valid_csv_bytes,
        file_name=full_path_valid,
        mime="text/csv",
    )


