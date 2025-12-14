import streamlit as st
import pandas as pd
import time
import requests
import re
import json

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
from openai import OpenAI


# =============================================================================
# 공통: max_tokens 안전 캡
# =============================================================================
def calc_max_tokens(batch_len: int, tokens_per_line: int = 160, cap: int = 12000) -> int:
    if batch_len <= 0:
        return 1000
    return max(256, min(cap, tokens_per_line * batch_len))


# =============================================================================
# 0) address_info 로드 및 컬럼 설정
# =============================================================================
ADDRESS_INFO_PATH = os.path.join("data", "address_info_2025_1103.xlsx")
address_info = pd.read_excel(ADDRESS_INFO_PATH)

SIDO_COL = "시도명"
SIGUNGU_COL = "시군구명"
EUP_COL = "읍면동명" if "읍면동명" in address_info.columns else None
BEOP_COL = "법정동명" if "법정동명" in address_info.columns else None

VALID_SIDO = [
    '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시',
    '세종특별자치시', '경기도', '충청북도', '충청남도', '전라남도', '경상북도', '경상남도',
    '제주특별자치도', '강원특별자치도', '전북특별자치도'
]

SIDO_ALIAS_MAP = {
    "서울": "서울특별시", "서울시": "서울특별시", "서울특별시": "서울특별시",
    "부산": "부산광역시", "부산시": "부산광역시", "부산광역시": "부산광역시",
    "대구": "대구광역시", "대구시": "대구광역시", "대구광역시": "대구광역시",
    "인천": "인천광역시", "인천시": "인천광역시", "인천광역시": "인천광역시",
    "광주": "광주광역시", "광주시": "광주광역시", "광주광역시": "광주광역시",
    "대전": "대전광역시", "대전시": "대전광역시", "대전광역시": "대전광역시",
    "울산": "울산광역시", "울산시": "울산광역시", "울산광역시": "울산광역시",
    "세종": "세종특별자치시", "세종시": "세종특별자치시", "세종특별자치시": "세종특별자치시",
    "경기": "경기도", "경기도": "경기도",
    "충북": "충청북도", "충청북도": "충청북도",
    "충남": "충청남도", "충청남도": "충청남도",
    "전남": "전라남도", "전라남도": "전라남도",
    "경북": "경상북도", "경상북도": "경상북도",
    "경남": "경상남도", "경상남도": "경상남도",
    "제주": "제주특별자치도", "제주도": "제주특별자치도", "제주특별자치도": "제주특별자치도",
    "강원": "강원특별자치도", "강원도": "강원특별자치도", "강원특별자치도": "강원특별자치도",
    "전북": "전북특별자치도", "전라북도": "전북특별자치도", "전북특별자치도": "전북특별자치도",
}


def _norm_key(s: str) -> str:
    if s is None:
        return ""
    x = str(s)
    x = re.sub(r"\(.*?\)", " ", x)
    x = x.replace(",", " ")
    x = re.sub(r"\s+", "", x)
    return x.strip()


# =============================================================================
# 붙여쓰기 공백 복원(카카오/추출 안정성)
# =============================================================================
SIDO_LONG_FOR_SPACING = VALID_SIDO[:]
SIDO_SHORT_FOR_SPACING = ["서울","부산","대구","인천","광주","대전","울산","세종","경기","충북","충남","전남","경북","경남","제주","강원","전북"]

def restore_spacing_for_concatenated_addr(s: str) -> str:
    if not s or not isinstance(s, str):
        return s
    x = s.strip()
    if not x:
        return x

    for sd in sorted(SIDO_LONG_FOR_SPACING, key=len, reverse=True):
        x = re.sub(rf"^{re.escape(sd)}(?=[가-힣]+(시|군|구))", sd + " ", x)
    for sd in sorted(SIDO_SHORT_FOR_SPACING, key=len, reverse=True):
        x = re.sub(rf"^{re.escape(sd)}(?=[가-힣]+(시|군|구))", sd + " ", x)

    x = re.sub(r"([가-힣]+시)(?=[가-힣])", r"\1 ", x)
    x = re.sub(r"([가-힣]+군)(?=[가-힣])", r"\1 ", x)
    x = re.sub(r"([가-힣]+구)(?=[가-힣])", r"\1 ", x)
    x = re.sub(r"([가-힣]+(면|읍))(?=[가-힣])", r"\1 ", x)

    x = re.sub(r"\s+", " ", x).strip()
    return x


# =============================================================================
# (추가) 시군구 기반 시도 추론(시도 누락 케이스 보완)
# =============================================================================
SIGUNGU_TO_SIDO_COUNTS = {}
SIGUNGU_UNIQUE_LIST = []
SIDO_TO_SIGUNGU_LIST = {}
SIDO_TO_SIGUNGU_MAP = {}

SIDO_TO_EUP_LIST = {}
SIDO_TO_EUP_MAP = {}
SIDO_TO_BEOP_LIST = {}
SIDO_TO_BEOP_MAP = {}

if (SIDO_COL in address_info.columns) and (SIGUNGU_COL in address_info.columns):
    tmp = address_info[[SIDO_COL, SIGUNGU_COL]].dropna().copy()
    tmp[SIDO_COL] = tmp[SIDO_COL].astype(str).str.strip()
    tmp[SIGUNGU_COL] = tmp[SIGUNGU_COL].astype(str).str.strip()

    grp = tmp.groupby([SIGUNGU_COL, SIDO_COL]).size().reset_index(name="cnt")
    for _, r in grp.iterrows():
        sg = r[SIGUNGU_COL]
        sd = r[SIDO_COL]
        cnt = int(r["cnt"])
        SIGUNGU_TO_SIDO_COUNTS.setdefault(sg, {})
        SIGUNGU_TO_SIDO_COUNTS[sg][sd] = SIGUNGU_TO_SIDO_COUNTS[sg].get(sd, 0) + cnt

    SIGUNGU_UNIQUE_LIST = sorted(tmp[SIGUNGU_COL].drop_duplicates().tolist(), key=lambda x: (-len(x), x))

    for sd, sub in address_info.groupby(SIDO_COL):
        sd = str(sd).strip()
        sg_list = sub[SIGUNGU_COL].dropna().astype(str).str.strip().drop_duplicates().tolist()
        SIDO_TO_SIGUNGU_LIST[sd] = sg_list
        SIDO_TO_SIGUNGU_MAP[sd] = {_norm_key(x): x for x in sg_list}

        if EUP_COL and (EUP_COL in sub.columns):
            eup_list = sub[EUP_COL].dropna().astype(str).str.strip().drop_duplicates().tolist()
            SIDO_TO_EUP_LIST[sd] = eup_list
            SIDO_TO_EUP_MAP[sd] = {_norm_key(x): x for x in eup_list}
        else:
            SIDO_TO_EUP_LIST[sd] = []
            SIDO_TO_EUP_MAP[sd] = {}

        if BEOP_COL and (BEOP_COL in sub.columns):
            beop_list = sub[BEOP_COL].dropna().astype(str).str.strip().drop_duplicates().tolist()
            SIDO_TO_BEOP_LIST[sd] = beop_list
            SIDO_TO_BEOP_MAP[sd] = {_norm_key(x): x for x in beop_list}
        else:
            SIDO_TO_BEOP_LIST[sd] = []
            SIDO_TO_BEOP_MAP[sd] = {}

def detect_and_normalize_sido(addr: str) -> str:
    if not addr or not isinstance(addr, str):
        return ""
    tokens = addr.replace(",", " ").split()
    for t in tokens[:3]:
        if t in SIDO_ALIAS_MAP:
            return SIDO_ALIAS_MAP[t]
        for k, v in SIDO_ALIAS_MAP.items():
            if k and k in t:
                return v
    return ""

def infer_sigungu_from_addr(addr: str) -> str:
    if not addr or not SIGUNGU_UNIQUE_LIST:
        return ""
    s = str(addr)
    tokens = s.replace(",", " ").split()
    token_set = set(tokens)
    for cand in SIGUNGU_UNIQUE_LIST:
        if cand in token_set:
            return cand
    best = ""
    best_pos = 10**9
    best_len = -1
    for cand in SIGUNGU_UNIQUE_LIST:
        pos = s.find(cand)
        if pos >= 0:
            c_len = len(cand)
            if (pos < best_pos) or (pos == best_pos and c_len > best_len):
                best_pos = pos
                best_len = c_len
                best = cand
    return best

def infer_sido_from_sigungu(sigungu: str) -> str:
    if not sigungu:
        return ""
    m = SIGUNGU_TO_SIDO_COUNTS.get(sigungu, {})
    if not m:
        return ""
    if len(m) == 1:
        return next(iter(m.keys()))
    return sorted(m.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

def infer_sido_when_missing(addr: str) -> str:
    sg = infer_sigungu_from_addr(addr)
    if sg:
        sd = infer_sido_from_sigungu(sg)
        if sd:
            return sd
    return ""


# =============================================================================
# 1단계(sec) - 검색용 주소 정제
# =============================================================================
GPT_ADDRESS_SYSTEM_PROMPT_step1 = """
당신은 주소 정제 전문가입니다.

목표:
- 지도/주소 검색이 잘 되도록 불필요한 요소만 제거합니다.
- (국화리), (이현동) 같은 괄호 안 표기는 유지하세요.

규칙:
1) 지번/건물번호가 여러개면 맨 처음 것만 남기세요.
2) 세부주소(층/호 등)는 제거하세요. 예: "103-2502", "302호", "1층" 등.
3) 주소 끝에 붙는 시설명/센터명 같은 꼬리 단어는 제거하세요.
4) 숫자/도로명/번지 및 시도/시군구/읍면동 자체를 임의 수정 금지.
5) 반드시 입력 순서대로, 한 줄에 1개 주소만 출력.
6) 다른 설명문/번호/불릿/코드펜스 금지.
7) 한 줄에 주소가 2개 이상 들어가면, 첫 번째 주소만 남기고 나머지는 삭제하세요.
""".strip()

def keep_first_address_only(s: str) -> str:
    if not s or not isinstance(s, str):
        return s
    x = s.strip()
    if "\n" in x:
        x = x.split("\n", 1)[0].strip()
    separators = [" / ", " /", "/ ", "/", " ; ", ";", " | ", "|", " · ", "•"]
    for sep in separators:
        if sep in x:
            x = x.split(sep, 1)[0].strip()
    if " , " in x:
        x = x.split(" , ", 1)[0].strip()
    return x

def clean_addresses_step1_sec_batch(
    df: pd.DataFrame,
    address_col: str,
    client: OpenAI,
    batch_size: int = 50,
    progress_hook=None,
    output_col: str = "주소_1단계",
) -> pd.DataFrame:
    addrs = df[address_col].astype(str).tolist()
    total = len(addrs)
    cleaned = [None] * total

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = addrs[start:end]
        batch_len = end - start

        user_content = "아래 주소들을 입력 순서대로, 한 줄에 하나씩 정제해서 반환하세요.\n"
        user_content += "\n".join(batch)

        max_tok = calc_max_tokens(batch_len=batch_len, tokens_per_line=120, cap=8000)

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": GPT_ADDRESS_SYSTEM_PROMPT_step1},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=max_tok,
        )

        text = (resp.choices[0].message.content or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for i in range(start, end):
            local_idx = i - start
            val = lines[local_idx] if local_idx < len(lines) else addrs[i]
            val = keep_first_address_only(val)
            val = restore_spacing_for_concatenated_addr(val)
            cleaned[i] = val

        if progress_hook is not None:
            progress_hook(end, total)

    out = df.copy()
    out[output_col] = cleaned
    return out


# =============================================================================
# 2단계: 카카오 API 조회(주소_1단계 기준) -> 기준주소(edit_address) 생성
# =============================================================================
BASE_URL = "https://dapi.kakao.com/v2/local/search/address.json"
session = requests.Session()

def search_address_once(query: str, page: int = 1, size: int = 10, timeout: float = 5.0):
    if not query or not isinstance(query, str):
        return None
    params = {"query": query, "page": page, "size": size}
    try:
        resp = session.get(BASE_URL, params=params, timeout=timeout)
        if resp.status_code == 429:
            return {"_error": "rate_limited", "_status": 429}
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"_error": str(e), "_status": getattr(e.response, "status_code", None)}

def parse_first_document(payload: dict):
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
    attempt = 0
    while True:
        payload = search_address_once(addr)
        if not payload or "_error" not in payload or payload.get("_status") not in (429, 500, 502, 503, 504):
            return parse_first_document(payload)
        if attempt >= max_retries:
            return parse_first_document(payload)
        time.sleep(base_sleep * (2 ** attempt))
        attempt += 1

def enrich_df_with_kakao_and_edit_address(
    df: pd.DataFrame,
    address_col_step1: str = "주소_1단계",
    progress_hook=None
) -> pd.DataFrame:
    """
    - 카카오 조회는 주소_1단계 기준
    - 조회 성공(좌표 또는 address_name 존재) 시 edit_address = kakao_address_name
    - 실패 시 edit_address = 주소_1단계
    """
    if address_col_step1 not in df.columns:
        raise ValueError(f"'{address_col_step1}' 컬럼 없음")

    total = len(df)
    records = []
    edit_list = []

    for i, addr in enumerate(df[address_col_step1].astype(str).tolist()):
        rec = query_with_backoff(addr)
        records.append(rec)

        kakao_name = (rec.get("kakao_address_name") or "").strip() if isinstance(rec, dict) else ""
        has_any = bool(kakao_name) or ((rec.get("kakao_x") is not None) and (rec.get("kakao_y") is not None))

        if has_any and kakao_name:
            edit_list.append(kakao_name)
        else:
            edit_list.append(addr)

        if progress_hook is not None:
            progress_hook(i + 1, total)

    res_df = pd.DataFrame(records)
    out = pd.concat([df.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)
    out["edit_address"] = edit_list
    return out


# =============================================================================
# 3단계: 최종 행정구역 추출(gpt_sido/gpt_sigungu/gpt_eupmyeondong)만 생성
#   - gpt_admin_fixed_addr 없음(요구사항)
#   - 읍면동 없으면: 법정동에서 찾아서 해당 법정동의 읍면동 매핑(가능할 때만)
# =============================================================================
STEP3_JSON_SCHEMA = {
    "name": "admin_extract_only_batch",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "idx": {"type": "integer"},
                        "addr_input": {"type": "string"},
                        "sido": {"type": "string"},
                        "sigungu": {"type": "string"},
                        "eupmyeondong": {"type": "string"},
                    },
                    "required": ["idx", "addr_input", "sido", "sigungu", "eupmyeondong"]
                }
            }
        },
        "required": ["items"]
    }
}

def build_final_edit_address(edit_addr: str, gpt_sido: str) -> str:
    """
    edit_address의 시도 표현을 표준 시도명으로 치환한 최종 주소 생성
    - gpt_sido가 있으면 최우선 사용
    - 없으면 SIDO_ALIAS_MAP으로 룰 기반 보정
    """
    if not edit_addr or not isinstance(edit_addr, str):
        return edit_addr

    addr = edit_addr.strip()

    # 1) gpt_sido가 있으면 주소 맨 앞 시도만 교체
    if gpt_sido:
        # 주소 맨 앞 토큰(공백 전까지)
        m = re.match(r"^(\S+)", addr)
        if m:
            first = m.group(1)
            # 첫 토큰이 시도 축약/표준 무엇이든 교체
            addr = re.sub(r"^\S+", gpt_sido, addr, count=1)
        return addr

    # 2) gpt_sido 없으면 alias map으로만 보정
    for short, full in SIDO_ALIAS_MAP.items():
        if addr.startswith(short + " "):
            return re.sub(r"^" + re.escape(short), full, addr, count=1)

    # 3) 시도 못 찾으면 그대로
    return addr


def _fallback_parse_items_json(raw: str):
    if not raw:
        return None
    txt = raw.strip()
    txt = txt.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
            return obj
    except Exception:
        return None
    return None

def build_step3_system_prompt() -> str:
    return f"""
당신은 대한민국 주소의 행정구역(시도/시군구/읍면동) 추출/정규화 전문가입니다.

중요:
- 출력은 반드시 JSON만 출력합니다. 설명문 금지.
- 주소에 없는 행정구역을 임의 생성하지 않습니다.
- 시도는 가능하면 다음 목록 중 하나로 정규화합니다: {VALID_SIDO}

추출 규칙:
- sido: 표준 시도명
- sigungu: 시/군/구
- eupmyeondong: 읍/면/동/리

보정 규칙(옵션):
- eupmyeondong이 비어 있으면:
  1) 주소에서 '법정동명'이 발견되면,
  2) 해당 법정동에 매핑되는 '읍면동명'이 address_info에 존재할 경우 eupmyeondong으로 채웁니다.
- 그래도 없으면 빈칸("")으로 둡니다.
""".strip()

def build_beop_to_eup_map(sido: str, sigungu: str = "") -> dict:
    if not BEOP_COL or not EUP_COL:
        return {}
    if (SIDO_COL not in address_info.columns) or (SIGUNGU_COL not in address_info.columns):
        return {}
    sub = address_info[address_info[SIDO_COL] == sido].copy()
    if sigungu:
        sub = sub[sub[SIGUNGU_COL] == sigungu].copy()
    sub = sub.dropna(subset=[BEOP_COL, EUP_COL])
    if sub.empty:
        return {}
    m = {}
    for _, r in sub[[BEOP_COL, EUP_COL]].iterrows():
        b = str(r[BEOP_COL]).strip()
        e = str(r[EUP_COL]).strip()
        if b and e and b not in m:
            m[b] = e
    return m

def find_first_beop_in_addr(addr: str, beop_candidates: list) -> str:
    if not addr or not beop_candidates:
        return ""
    s = str(addr)
    best = ""
    best_pos = 10**9
    for b in beop_candidates:
        if not b:
            continue
        pos = s.find(b)
        if pos >= 0 and pos < best_pos:
            best_pos = pos
            best = b
    return best

def postfill_eup_from_beop(sido: str, sigungu: str, addr: str, eup: str) -> str:
    if eup:
        return eup
    if not (sido and BEOP_COL and EUP_COL):
        return eup

    m = build_beop_to_eup_map(sido, sigungu)
    if not m:
        m = build_beop_to_eup_map(sido, "")

    if not m:
        return eup

    hit_beop = find_first_beop_in_addr(addr, list(m.keys()))
    if not hit_beop:
        return eup

    return m.get(hit_beop, "") or ""


def normalize_admin_step3_from_edit_address(
    df: pd.DataFrame,
    address_col: str = "edit_address",
    client: OpenAI = None,
    batch_size: int = 25,
    progress_hook=None,
    out_cols_prefix: str = "gpt_",
    debug_preview: bool = True,
) -> pd.DataFrame:
    if address_col not in df.columns:
        raise ValueError(f"'{address_col}' 컬럼 없음")

    addrs = df[address_col].astype(str).tolist()
    total = len(addrs)

    sido_out = [""] * total
    sigungu_out = [""] * total
    dong_out = [""] * total

    ok_cnt = 0
    used_fallback_cnt = 0
    first_bad = None

    sys_prompt = build_step3_system_prompt()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = addrs[start:end]
        batch_len = end - start

        # idx<TAB>addr
        user_lines = [f"{start + i}\t{a}" for i, a in enumerate(batch)]
        user_content = (
            "아래는 'idx<TAB>address' 목록입니다.\n"
            "각 idx에 대해 items에 동일 idx로 결과를 채우세요.\n"
            + "\n".join(user_lines)
        )

        max_tok = calc_max_tokens(batch_len=batch_len, tokens_per_line=220, cap=12000)

        obj = None
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                response_format={"type": "json_schema", "json_schema": STEP3_JSON_SCHEMA},
                max_tokens=max_tok,
            )
            raw = (resp.choices[0].message.content or "").strip()
            obj = json.loads(raw)
        except Exception as e:
            try:
                resp2 = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_content + "\n\n반드시 {'items':[...]} 형태의 JSON만 출력하세요."},
                    ],
                    temperature=0,
                    max_tokens=max_tok,
                )
                raw2 = (resp2.choices[0].message.content or "").strip()
                obj2 = _fallback_parse_items_json(raw2)
                if obj2 is not None:
                    obj = obj2
                    used_fallback_cnt += 1
                else:
                    if first_bad is None:
                        first_bad = f"배치 {start}-{end} 실패(Structured+fallback): {e}\nraw2(앞 800자): {raw2[:800]}"
            except Exception as e2:
                if first_bad is None:
                    first_bad = f"배치 {start}-{end} 실패(Structured 후 fallback 호출도 실패): {e2}"

        items = []
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            items = obj["items"]

        by_idx = {}
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("idx"), int):
                by_idx[it["idx"]] = it

        for i in range(start, end):
            addr0 = addrs[i]
            it = by_idx.get(i)

            if not it:
                # GPT 실패 시 최소 룰 기반(시도만이라도)
                sd = detect_and_normalize_sido(addr0) or infer_sido_when_missing(addr0)
                sido_out[i] = sd
                sigungu_out[i] = ""
                dong_out[i] = ""
                continue

            sd = (it.get("sido") or "").strip()
            sg = (it.get("sigungu") or "").strip()
            dn = (it.get("eupmyeondong") or "").strip()

            if not sd:
                sd = detect_and_normalize_sido(addr0) or infer_sido_when_missing(addr0)

            # 법정동->읍면동 매핑 보정(옵션)
            dn = postfill_eup_from_beop(sd, sg, addr0, dn)

            sido_out[i] = sd
            sigungu_out[i] = sg
            dong_out[i] = dn
            ok_cnt += 1

        if progress_hook is not None:
            progress_hook(end, total)

    out = df.copy()
    out[f"{out_cols_prefix}sido"] = sido_out
    out[f"{out_cols_prefix}sigungu"] = sigungu_out
    out[f"{out_cols_prefix}eupmyeondong"] = dong_out

    if debug_preview:
        rate = (ok_cnt / max(total, 1)) * 100
        st.info(f"[최종 행정구역 추출 성공률] {ok_cnt}/{total} ({rate:.1f}%)")
        if first_bad:
            with st.expander("최종 추출 디버그(첫 실패 메시지)"):
                st.text(first_bad)

    return out


# =============================================================================
# 4) 시도별 반경 그룹핑(컬럼명 고정, suffix 제거)
# =============================================================================
def group_by_radius_per_sido(
    df_in: pd.DataFrame,
    sido_col: str = "gpt_sido",
    lat_col: str = "kakao_y",
    lon_col: str = "kakao_x",
    sido_radius_map: dict = None,
    default_radius_km: float = 10.0,
):
    out = df_in.copy()

    out["component_id"] = np.nan
    out["component_size"] = np.nan
    out["is_isolated"] = np.nan
    out["group_key"] = ""
    out["radius_km_used"] = np.nan

    if sido_radius_map is None:
        sido_radius_map = {}

    work = out.dropna(subset=[lat_col, lon_col]).copy()
    if work.empty:
        return out, pd.DataFrame()

    work[sido_col] = work[sido_col].astype(str).str.strip()

    summaries = []
    for sd, gsd in work.groupby(sido_col):
        if not sd or sd == "nan":
            radius_km = float(default_radius_km)
            sd_key = "미상"
        else:
            radius_km = float(sido_radius_map.get(sd, default_radius_km))
            sd_key = sd

        coords_rad = np.radians(gsd[[lat_col, lon_col]].to_numpy())
        D_rad = haversine_distances(coords_rad)
        D_km = D_rad * 6371.0

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

        sd_prefix = re.sub(r"[^0-9A-Za-z가-힣]+", "", sd_key) or "SIDO"
        idx_map = gsd.index.to_numpy()

        group_keys = np.where(
            isolated,
            np.array([f"{sd_prefix}_I{ix}" for ix in idx_map]),
            np.array([f"{sd_prefix}_C{c}" for c in labels])
        )

        out.loc[idx_map, "component_id"] = labels
        out.loc[idx_map, "component_size"] = size_arr
        out.loc[idx_map, "is_isolated"] = isolated
        out.loc[idx_map, "group_key"] = group_keys
        out.loc[idx_map, "radius_km_used"] = radius_km

        tmp = (
            out.loc[idx_map, ["group_key", "component_id", "component_size", lat_col, lon_col]]
            .assign(_one=1, sido=sd_key, radius_km=radius_km)
            .groupby(["sido", "radius_km", "group_key", "component_id", "component_size"], as_index=False)
            .agg(
                n_points=("_one", "sum"),
                center_lat=(lat_col, "mean"),
                center_lon=(lon_col, "mean"),
            )
            .sort_values(["n_points", "group_key"], ascending=[False, True])
        )
        summaries.append(tmp)

    summary_df = pd.concat(summaries, axis=0, ignore_index=True) if summaries else pd.DataFrame()
    return out, summary_df


# =============================================================================
# 5) Streamlit UI
# =============================================================================
st.title("주소 정제 → 카카오 조회 → (카카오명/1단계주소) 기반 행정구역 추출 → 시도별 반경 그룹핑")

st.markdown("#### 1. API 키 입력")
gpt_api_key = st.text_input("OpenAI GPT API 키를 입력하세요", type="password")
kakao_api_key = st.text_input("Kakao REST API 키를 입력하세요", type="password")
st.caption("※ Kakao 로컬 API는 계정 기준 월 최대 3,000,000건 조회 제한이 있으니 호출량에 유의해 주세요.\n사용량은 kakao developers 내 쿼터 참고")

st.markdown("#### 2. 분석 옵션 설정(세부)")
st.write("- 기본 반경(km)을 먼저 설정 후, 시도별로 필요한 경우만 조정")

default_radius_km = st.number_input(
    "기본 반경(km) (먼저 설정)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.5,
    key="default_radius_km"
)

# 최초 1회: 시도별 기본값/위젯 키 초기화
if "sido_radius_map" not in st.session_state:
    st.session_state["sido_radius_map"] = {}
    for sd in VALID_SIDO:
        st.session_state["sido_radius_map"][sd] = float(default_radius_km)
        st.session_state[f"radius_{sd}"] = float(default_radius_km)  # ★ 위젯 key도 같이 초기화

col_a, col_b = st.columns([1, 3])
with col_a:
    apply_default = st.button("기본값을 전체 시도에 일괄 적용")
with col_b:
    st.caption("※ 기본값 변경 후, 버튼을 누르면 모든 시도 값이 기본값으로 리셋됨")

if apply_default:
    for sd in VALID_SIDO:
        st.session_state["sido_radius_map"][sd] = float(default_radius_km)
        st.session_state[f"radius_{sd}"] = float(default_radius_km)  # ★ 핵심
    st.rerun()  

with st.expander("시도별 반경 조정(필요한 시도만 변경)"):
    for sd in VALID_SIDO:
        # value=는 사실상 의미 약함(키가 이미 있으면 session_state 우선)
        v = st.number_input(
            f"{sd} 반경(km)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["sido_radius_map"].get(sd, default_radius_km)),
            step=0.5,
            key=f"radius_{sd}"
        )
        # 위젯 값 -> map 동기화
        st.session_state["sido_radius_map"][sd] = float(v)

sido_radius_map = dict(st.session_state["sido_radius_map"])

min_group_size = st.number_input("유효 집단 최소 건수 (기본 20)", min_value=1, value=20, step=1)

st.markdown("#### 3. 엑셀 파일 업로드")
uploaded_file = st.file_uploader("그루핑 대상 엑셀 파일을 업로드하세요", type=["xlsx"])
sheet_selected = None
row_skips = st.number_input("상단 스킵할 행 수(skiprows)", min_value=0, value=0, step=1)

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        xls = pd.ExcelFile(uploaded_file)
        sheet_selected = st.selectbox("시트 선택", options=xls.sheet_names, index=0)
    except Exception as e:
        st.warning(f"시트 목록을 불러오는 중 오류 발생: {e}")
        sheet_selected = None

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
    if sheet_selected is None:
        st.error("시트를 선택할 수 없습니다. 엑셀 파일을 확인해 주세요.")
        st.stop()

    gpt_client = OpenAI(api_key=gpt_api_key)
    session.headers.update({"Authorization": f"KakaoAK {kakao_api_key}"})

    try:
        uploaded_file.seek(0)
        df_origin = pd.read_excel(uploaded_file, sheet_name=sheet_selected, skiprows=int(row_skips))
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류 발생: {e}")
        st.stop()

    df = df_origin

    st.write("원본 데이터 preview:")
    st.dataframe(df.head())

    st.markdown("#### 4. 주소 처리: 1단계 → 카카오 조회 → edit_address 생성 → 최종 행정구역 추출")

    gpt_progress = st.progress(0)
    gpt_status = st.empty()

    def progress_hook(done: int, total: int, label: str):
        if total <= 0:
            return
        ratio = done / total
        gpt_progress.progress(ratio)
        gpt_status.text(f"{label}: {done}/{total} ({ratio*100:.1f}%)")

    # -------------------------
    # (1) 1단계(sec)
    # -------------------------
    gpt_progress = st.progress(0)
    gpt_status = st.empty()

    with st.spinner("1단계(주소 정제) 수행 중..."):
        def step1_hook(done, total):
            progress_hook(done, total, "1단계")
        df_step1 = clean_addresses_step1_sec_batch(
            df,
            address_col="주소",
            client=gpt_client,
            progress_hook=step1_hook,
            output_col="주소_1단계",
            batch_size=50,
        )

    st.success("1단계 완료")
    st.dataframe(df_step1[["주소", "주소_1단계"]].head())

    # -------------------------
    # (2) 카카오 조회 + edit_address 생성
    # -------------------------
    gpt_progress = st.progress(0)
    gpt_status = st.empty()

    with st.spinner("2단계(카카오 주소 조회 + edit_address 생성) 수행 중..."):
        def kakao_hook(done, total):
            progress_hook(done, total, "카카오조회")
        df_kakao = enrich_df_with_kakao_and_edit_address(
            df_step1,
            address_col_step1="주소_1단계",
            progress_hook=kakao_hook
        )

    st.success("카카오 조회 완료")
    st.dataframe(df_kakao[["주소", "주소_1단계", "kakao_address_name", "kakao_address_type", "edit_address"]].head())

    # -------------------------
    # (3) 최종 행정구역 추출(edit_address 기준)
    # -------------------------
    gpt_progress = st.progress(0)
    gpt_status = st.empty()

    with st.spinner("3단계(최종 행정구역 추출) 수행 중..."):
        def step3_hook(done, total):
            progress_hook(done, total, "최종추출")
        df_final = normalize_admin_step3_from_edit_address(
            df_kakao,
            address_col="edit_address",
            client=gpt_client,
            progress_hook=step3_hook,
            out_cols_prefix="gpt_",
            batch_size=25,
            debug_preview=True,
        )
        df_final["final_address"] = df_final.apply(
        lambda r: build_final_edit_address(
            edit_addr=r.get("edit_address", ""),
            gpt_sido=(str(r.get("gpt_sido", "")).strip() if pd.notna(r.get("gpt_sido", "")) else "")
        ),
        axis=1
    )

    st.success("최종 행정구역 추출 완료")
    st.dataframe(df_final[["주소", "주소_1단계", "edit_address", "gpt_sido", "gpt_sigungu", "gpt_eupmyeondong"]].head())

    # ------------------------------------------------
    # 좌표 숫자형 변환
    # ------------------------------------------------
    df_valid = df_final.copy()
    df_valid["kakao_x"] = pd.to_numeric(df_valid["kakao_x"], errors="coerce")
    df_valid["kakao_y"] = pd.to_numeric(df_valid["kakao_y"], errors="coerce")

    # ------------------------------------------------
    # 시도별 반경 그룹핑
    # ------------------------------------------------
    st.markdown("#### 5. 시도별 반경 그룹핑 실행")

    with st.spinner("시도별 거리 기반 클러스터링 중..."):
        labeled_df, summary = group_by_radius_per_sido(
            df_valid,
            sido_col="gpt_sido",
            lat_col="kakao_y",
            lon_col="kakao_x",
            sido_radius_map=sido_radius_map,
            default_radius_km=float(default_radius_km),
        )

    st.success("시도별 그룹핑 완료")
    st.write("그룹 요약 정보 일부:")
    st.dataframe(summary.head())

    # ------------------------------------------------
    # 최소 건수 이상 집단 필터링
    # ------------------------------------------------
    st.markdown("#### 6. 최소 건수 이상 집단 필터링")
    vc = labeled_df["group_key"].value_counts()
    valid_keys = vc[vc >= min_group_size].index
    labeled_df_sec = labeled_df.loc[labeled_df["group_key"].isin(valid_keys)].copy()

    st.write(f"최소 {min_group_size}건 이상 집단 수: {labeled_df_sec['group_key'].nunique()}개")
    st.dataframe(labeled_df_sec.head())

    # ------------------------------------------------
    # 파일 저장(고정 파일명)
    # ------------------------------------------------
    full_path_all = "그룹핑_최종완료_전체.csv"
    full_path_valid = f"그룹핑_최종완료_{min_group_size}개이상.csv"

    try:
        labeled_df.to_csv(full_path_all, encoding="euc-kr", index=False)
        labeled_df_sec.to_csv(full_path_valid, encoding="euc-kr", index=False)
        st.info(f"로컬 저장:\n- {full_path_all}\n- {full_path_valid}")
    except Exception as e:
        st.warning(f"로컬 저장 오류(다운로드는 가능): {e}")

    all_csv_bytes = labeled_df.to_csv(index=False, encoding="euc-kr").encode("euc-kr", errors="ignore")
    valid_csv_bytes = labeled_df_sec.to_csv(index=False, encoding="euc-kr").encode("euc-kr", errors="ignore")

    st.download_button(
        label="전체 결과 다운로드",
        data=all_csv_bytes,
        file_name=full_path_all,
        mime="text/csv",
    )

    st.download_button(
        label=f"최소 {min_group_size}건 이상 집단 결과 다운로드",
        data=valid_csv_bytes,
        file_name=full_path_valid,
        mime="text/csv",
    )
