from __future__ import annotations
import io
from pathlib import Path
import librosa
import numpy as np
import streamlit as st
import torch
import yaml
from src.audio import log_mel, mfcc
from src.models import build_model

ROOT = Path(__file__).resolve().parent
LABELS = ["angry", "fearful", "happy", "neutral", "sad"]
LABEL_AR = {"angry": "غاضب", "fearful": "خائف", "happy": "سعيد", "neutral": "محايد", "sad": "حزين"}
MODEL_NAMES = {"CNN": "cnn", "CNN + BiLSTM + Attention": "cnn_bilstm_attention", "LSTM": "lstm", "GRU": "gru", "MLP": "mlp"}

st.set_page_config(page_title="YemenAI | تحليل المشاعر الصوتية", page_icon="🎙️", layout="wide")

@st.cache_data
def read_config():
    with (ROOT / "configs" / "default.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_resource(show_spinner="جاري تحميل النموذج...")
def load_model(model_name: str, checkpoint_bytes: bytes, device_name: str):
    device = torch.device(device_name)
    try:
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = build_model(model_name, num_classes=len(LABELS))
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint

def prepare_features(audio_bytes: bytes, model_name: str, cfg: dict):
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=int(cfg["sample_rate"]), mono=True)
    if audio.size == 0:
        raise ValueError("الملف الصوتي فارغ أو غير قابل للقراءة.")
    if model_name in ("cnn", "cnn_bilstm_attention"):
        x = log_mel(audio, sample_rate=int(cfg["sample_rate"]), n_mels=int(cfg["n_mels"]), n_fft=int(cfg["n_fft"]), hop_length=int(cfg["hop_length"]))
        x = (x - x.mean()) / max(float(x.std()), 1e-6)
        tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    else:
        x = mfcc(audio, sample_rate=int(cfg["sample_rate"]), n_mfcc=int(cfg["n_mfcc"]), n_fft=int(cfg["n_fft"]), hop_length=int(cfg["hop_length"])).T
        x = (x - x.mean()) / max(float(x.std()), 1e-6)
        tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    return tensor, len(audio) / float(cfg["sample_rate"])

def main():
    cfg = read_config()
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {font-family:'Cairo',sans-serif}
    .block-container{padding-top:2rem;max-width:1200px}
    .hero{padding:1.7rem 2rem;border-radius:22px;background:linear-gradient(115deg,#102a43,#145374 58%,#168aad);color:white;margin-bottom:1.4rem}
    .hero h1{color:white;font-size:2rem;margin:0 0 .4rem}
    .hero p{color:#e4f3f8;margin:0}
    .emotion-card{border:1px solid #dbe7ef;border-radius:18px;padding:1.2rem;background:#f7fbfd;text-align:center}
    .emotion-name{font-size:1.8rem;font-weight:800;color:#12344d}
    .muted{color:#61798b;font-size:.9rem}
    [data-testid="stMetric"]{background:#f6fafc;border:1px solid #e2edf3;padding:12px 16px;border-radius:14px}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="hero" dir="rtl">
      <h1>🎙️ YemenAI — نظام تحليل المشاعر الصوتية</h1>
      <p>ارفع تسجيلاً صوتياً، واختر النموذج المدرّب، ثم اعرض نتيجة تصنيف المشاعر الخمسة ودرجات الثقة.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## إعدادات التحليل")
        model_label = st.selectbox("نموذج الذكاء الاصطناعي", list(MODEL_NAMES.keys()))
        model_name = MODEL_NAMES[model_label]
        st.caption("يجب أن يتوافق ملف checkpoint مع معمارية النموذج المختار.")
        checkpoint_file = st.file_uploader("ملف النموذج المدرّب (.pt / .pth)", type=["pt", "pth"], help="ارفع ملف best.pt الناتج من scripts/train.py.")
        st.divider()
        st.markdown("### إعدادات الصوت")
        st.write(f"معدل العينة: **{cfg['sample_rate']} Hz**")
        st.write(f"عدد فئات المشاعر: **{len(LABELS)}**")
        st.write("الميزات: **Log-Mel** أو **MFCC** حسب النموذج")
        st.divider()
        st.caption("YemenAI • Audio Emotion Recognition")

    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("### 1. أضف التسجيل الصوتي")
        audio_file = st.file_uploader("اختر ملفاً صوتياً", type=["wav", "mp3", "ogg", "flac", "m4a"], help="يفضل استخدام تسجيل واضح وخالٍ قدر الإمكان من الضوضاء.", key="audio_upload")
        if audio_file:
            st.audio(audio_file.getvalue(), format=audio_file.type)
            st.caption(f"الملف المحدد: {audio_file.name}")
    with right:
        st.markdown("### 2. تشغيل التحليل")
        st.info("ارفع ملف النموذج المدرّب والتسجيل الصوتي، ثم اضغط على زر التحليل.")
        run = st.button("🔍 تحليل المشاعر", type="primary", use_container_width=True, disabled=(audio_file is None or checkpoint_file is None))

    if run:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            with st.spinner("يتم تجهيز الصوت وتشغيل النموذج..."):
                model, checkpoint = load_model(model_name, checkpoint_file.getvalue(), device_name)
                features, duration = prepare_features(audio_file.getvalue(), model_name, cfg)
                with torch.inference_mode():
                    logits = model(features.to(device_name))
                    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            best_index = int(np.argmax(probabilities))
            best_label = LABELS[best_index]
            confidence = float(probabilities[best_index]) * 100
            st.divider()
            st.markdown("## نتيجة التحليل")
            result_col, info_col = st.columns([1.1, 1], gap="large")
            with result_col:
                st.markdown(f"""
                <div class="emotion-card" dir="rtl">
                  <div class="muted">المشاعر الأكثر احتمالاً</div>
                  <div class="emotion-name">{LABEL_AR[best_label]}</div>
                  <div class="muted">{best_label.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
            with info_col:
                st.metric("درجة ثقة النموذج", f"{confidence:.1f}%")
                st.metric("مدة التسجيل", f"{duration:.2f} ثانية")
                st.metric("جهاز التنفيذ", "GPU" if device_name == "cuda" else "CPU")
            st.markdown("### درجات الثقة لجميع المشاعر")
            st.bar_chart({"المشاعر": [LABEL_AR[x] for x in LABELS], "درجة الثقة (%)": [round(float(p)*100, 2) for p in probabilities]}, x="المشاعر", y="درجة الثقة (%)", horizontal=True, use_container_width=True)
            with st.expander("تفاصيل تقنية"):
                st.write(f"النموذج: {model_name}")
                st.write(f"شكل مدخل النموذج: {tuple(features.shape)}")
                st.write(f"ملف النموذج: {checkpoint_file.name}")
                if isinstance(checkpoint, dict) and "epoch" in checkpoint:
                    st.write(f"أفضل Epoch محفوظ: {checkpoint['epoch']}")
                if isinstance(checkpoint, dict) and "best_validation_macro_f1" in checkpoint:
                    st.write(f"أفضل Validation Macro-F1: {float(checkpoint['best_validation_macro_f1']):.4f}")
            st.warning("هذه النتيجة تقدير آلي مبني على النموذج المدرّب، وليست حكماً مؤكداً على الحالة النفسية للمتحدث. تتأثر الدقة بجودة التسجيل والبيانات والتدريب.")
        except Exception as exc:
            st.error(f"تعذر إكمال التحليل: {exc}")
            st.info("تحقق من توافق النموذج المختار مع ملف checkpoint، ومن أن الملف يحتوي على model_state وأن النموذج قد تم تدريبه مسبقاً.")

    st.divider()
    st.markdown("<div class='muted' dir='rtl' style='text-align:center'>YemenAI — مشروع بحثي للتعرف على المشاعر من الكلام | غاضب، خائف، سعيد، محايد، حزين</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
