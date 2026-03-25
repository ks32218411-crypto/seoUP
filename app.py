import streamlit as st
import cv2
import numpy as np
import requests
import os
import io
from PIL import Image

# 웹페이지 기본 설정
st.set_page_config(page_title="AI 4K Upscaler", page_icon="✨", layout="wide")

MODEL_PATH = "EDSR_x4.pb"
MODEL_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"

@st.cache_resource
def download_model():
    """웹사이트 최초 접속 시, 38MB짜리 인공지능 해상도 복원 뇌를 다운받습니다."""
    if not os.path.exists(MODEL_PATH):
        # 다운로드가 안 된 상태라면 받기
        with st.spinner("🚀 최초 1회, AI 엔진(EDSR 4K 모델)을 조립 중입니다... (10초 소요)"):
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return True

@st.cache_resource
def load_ai():
    """다운받은 뇌를 장착하고 4배수(x4) 업스케일링 모드로 부팅합니다."""
    download_model()
    # 파이토치가 아닌, 순수 C++ 기반인 가장 가볍고 강력한 OpenCV DNN 엔진 사용
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_PATH)
    sr.setModel("edsr", 4) # 4배(4x) 화질 리마스터링 스위치 ON
    return sr

# ----------------- UI 렌더링 ----------------- #
st.title("✨ AI 4배 고화질 복원기 (Super Resolution)")
st.markdown("""
찌그러지고 화질구지가 된 작은 사진을 올리시면, 인공지능(EDSR 신경망)이 깨진 픽셀 사이사이를 상상해서 채워넣어 
**해상도를 4배로 뻥튀기(Upscaling)** 시켜드립니다! (예: 500픽셀 -> 2000픽셀 초고화질 😲)
*(⚠️ 이미지 크기가 너무 거대하면 무료 서버 CPU가 연산하다 뻗을 수도 있습니다. 작은 짤방/캐릭터 그림용으로 쓰세요!)*
""")

# ================= Sidebar: 고급 세부 설정 =================
st.sidebar.header("🎛️ AI 미세조정 (Finetuning)")
denoise_level = st.sidebar.slider("🧹 노이즈 제거 (잡티 지우기)", min_value=0.0, max_value=10.0, value=3.0, step=0.5, 
                                  help="수치를 올리면 옛날 사진의 자글자글한 노이즈가 부드럽게 밀립니다. (단점: 너무 올리면 수채화처럼 뭉개짐)")

sharpen_level = st.sidebar.slider("🔪 선명도 (샤프니스)", min_value=0.0, max_value=2.0, value=0.5, step=0.1,
                                  help="수치를 올리면 흐릿한 윤곽선이 칼처럼 날카로워집니다. (단점: 너무 올리면 픽셀이 깨져 보임)")

st.sidebar.markdown("---")
st.sidebar.info("💡 **추천 셋팅**: 인물/풍경 사진은 디노이즈를 살짝 주고(2.0), 애니메이션/글씨는 샤프니스(1.0)를 주면 좋습니다.")
# =========================================================

# 백그라운드 엔진 점화
try:
    sr = load_ai()
except Exception as e:
    st.error(f"엔진 점화 실패! 모델 다운로드 중에 네트워크가 끊겼을 수 있습니다. 새로고침 하세요: {e}")
    st.stop()

uploaded_file = st.file_uploader("📥 여기에 흐릿한 이미지를 마우스로 끌어다 놓으세요 (JPG/PNG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 1. 브라우저에서 올린 파일을 파이썬이 읽을 수 있게 변환 (Numpy Array 분해)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1) # OpenCV가 좋아하는 파랑-초록-빨강(BGR) 포맷
    
    # 원본 세로(h), 가로(w) 길이 추출
    orig_h, orig_w = image_bgr.shape[:2]
    
    # --- [추가] 실시간 프리뷰 생성 (0.1초 만에 가벼운 윤곽선 필터만 먼저 씌워보기) ---
    preview_bgr = image_bgr.copy()
    if denoise_level > 0:
        preview_bgr = cv2.fastNlMeansDenoisingColored(preview_bgr, None, denoise_level, denoise_level, 7, 21)
    if sharpen_level > 0:
        kernel = np.array([
            [0, -1, 0], 
            [-1, 5, -1], 
            [0, -1, 0]
        ])
        hard_sharpened = cv2.filter2D(preview_bgr, -1, kernel)
        preview_bgr = cv2.addWeighted(preview_bgr, 1.0 - (sharpen_level / 2.0), hard_sharpened, (sharpen_level / 2.0), 0)
        
    # 웹에서 예쁘게 보여주기 위해 레드-그린-블루(RGB) 포맷으로 최종 변환
    preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    # -------------------------------------------------------------
    
    # 3. 화면을 두 개의 기둥(Column)으로 좌우 분할해서 Before & After 연출
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"👀 실시간 필터 프리뷰 (`{orig_w} x {orig_h}`)")
        st.caption("✅ 1초(왼쪽 패널): 슬라이더를 움직이면 0.1초 만에 효과가 덧씌워진 원본 크기를 보여줍니다.")
        # use_container_width=True 로 하면 브라우저 폭에 맞춰 꽉 차게 띄워줌
        st.image(preview_rgb, use_container_width=True)
        
    st.divider()
    
    # 4. 버튼을 만들어두고 유저가 허락할 때 엄청 무거운 4배수 AI 연산 시작
    start_btn = st.button("🔥 15초(오른쪽 패널): 미친 화질로 4배 AI 복원 시작!", type="primary")
    
    if start_btn:
        # 안전장치: 이미 원본이 1000픽셀이 넘어가면 무료 서버에서 터질 확률이 높음
        if orig_w * orig_h > 1500 * 1500:
            st.warning("경고: 파일이 꽤 큰 편이라, 무료 CPU 등급에서는 화질 복원에 수 분 이상이 소요될 수 있습니다. 커피 한 잔 하고 오세요 ☕")
            
        with st.spinner("🧠 AI가 깨진 픽셀을 정밀하게 다시 채워넣고 있습니다... (화면 끄지 마세요)"):
            try:
                # [스텝 1] : 업스케일링 전 노이즈 제거 (가장 빠르고 효율적인 순서)
                if denoise_level > 0:
                    processed_bgr = cv2.fastNlMeansDenoisingColored(image_bgr, None, denoise_level, denoise_level, 7, 21)
                else:
                    processed_bgr = image_bgr
                    
                # [스텝 2] : 매직 시작! (AI가 이미지를 통째로 다시 그림)
                result_bgr = sr.upsample(processed_bgr)
                
                # [스텝 3] : 업스케일링 후 샤프닝 커널 덮어씌우기 (선명도 극대화)
                if sharpen_level > 0:
                    # 샤프닝을 위한 OpenCV 필터 행렬 (Convolution Kernel)
                    kernel = np.array([
                        [0, -1, 0], 
                        [-1, 5, -1], 
                        [0, -1, 0]
                    ])
                    # 필터 적용된 강력한 날카로운 이미지 생성
                    hard_sharpened = cv2.filter2D(result_bgr, -1, kernel)
                    # 유저가 설정한 슬라이더 수치(Level)만큼 원본과 날카로운 이미지를 비율 섞기 (블렌딩)
                    result_bgr = cv2.addWeighted(result_bgr, 1.0 - (sharpen_level / 2.0), hard_sharpened, (sharpen_level / 2.0), 0)
                
                # 결과물을 다시 모니터용 RGB로 변환
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                new_h, new_w = result_rgb.shape[:2]
                
                with col2:
                    st.subheader(f"✨ 4K AI 복원 완료 (`{new_w} x {new_h}`)")
                    st.image(result_rgb, use_container_width=True)
                    
                st.success(f"🎉 성공했습니다!! 원본보다 [해상도가 400% 상승] 했습니다. 아래 버튼을 눌러 결과물을 다운로드 하세요 🚀")
                
                # 6. 브라우저에서 '다운로드 버튼'을 띄우기 위해 그림 데이터를 다시 압축(Encoding)
                is_success, buffer = cv2.imencode(".png", result_bgr)
                io_buf = io.BytesIO(buffer)
                
                st.download_button(
                    label="📥 4K 고화질 압축 해제 무손실 이미지 다운로드",
                    data=io_buf,
                    file_name="AI_Upscaled_Result.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error("앗! 사진이 너무 커서 연산 중에 서버 메모리가 폭발했습니다 😂 더 작은 이미지를 올려주세요.")
