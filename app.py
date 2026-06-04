import streamlit as st
import numpy as np
import pickle
import json
import re
import os

@st.cache_resource
def load_model_and_config():
    import keras
    base_path = os.path.dirname(os.path.abspath(__file__))
    model = keras.models.load_model(os.path.join(base_path, 'lstm_model.keras'))
    with open(os.path.join(base_path, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(base_path, 'model_config.json'), 'r') as f:
        config = json.load(f)
    return model, tokenizer, config

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text, model, tokenizer, config):
    from keras.preprocessing.sequence import pad_sequences
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=config['MAX_LEN'], padding='post', truncating='post')
    pred_proba = model.predict(padded, verbose=0)
    label_classes = config['label_classes']
    num_classes = config['num_classes']
    if num_classes > 2:
        pred_idx = int(np.argmax(pred_proba, axis=1)[0])
        confidence = float(pred_proba[0][pred_idx])
        all_probs = {str(label_classes[i]): float(pred_proba[0][i]) for i in range(num_classes)}
    else:
        prob = float(pred_proba.flatten()[0])
        pred_idx = 1 if prob > 0.5 else 0
        confidence = prob if pred_idx == 1 else 1 - prob
        all_probs = {str(label_classes[0]): 1 - prob, str(label_classes[1]): prob}
    return str(label_classes[pred_idx]), confidence, all_probs

# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Sentiment Analysis LSTM",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.result-card {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
    margin: 1rem 0;
}
.positive { background: #d1fae5; color: #065f46; border: 2px solid #34d399; }
.negative { background: #fee2e2; color: #991b1b; border: 2px solid #f87171; }
.neutral  { background: #e0f2fe; color: #0369a1; border: 2px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Sentiment Analysis — Bidirectional LSTM")
st.caption("Text Mining | Tugas Machine Learning")

# ===================== LOAD MODEL =====================

with st.spinner("Loading model..."):
    model, tokenizer, config = load_model_and_config()

# ===================== SIDEBAR =====================

with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.metric("Test Accuracy", f"{config['test_accuracy']*100:.1f}%")
    st.metric("F1 Score", f"{config['f1_score']:.4f}")
    st.markdown("---")
    st.markdown("### ⚙️ Konfigurasi")
    st.write(f"**Vocab size:** {config['MAX_WORDS']:,}")
    st.write(f"**Max length:** {config['MAX_LEN']} tokens")
    st.write(f"**Embedding dim:** {config['EMBEDDING_DIM']}")
    st.write(f"**Kelas:** {', '.join([str(c) for c in config['label_classes']])}")
    st.markdown("---")
    base = os.path.dirname(os.path.abspath(__file__))
    for fname, title in [('training_history.png', 'Training History'), ('confusion_matrix.png', 'Confusion Matrix')]:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            st.markdown(f"### {title}")
            st.image(fpath)

# ===================== MAIN =====================

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### ✍️ Masukkan Teks")
    user_text = st.text_area(
        "Ketik atau paste teks di sini:",
        placeholder="Contoh: Produk ini sangat bagus, pengiriman cepat!",
        height=150,
        key="main_input"
    )

    st.markdown("**Coba contoh:**")
    c1, c2, c3 = st.columns(3)
    examples = [
        ("😊 Positif", "Produk sangat memuaskan, kualitas terbaik dan harga terjangkau!"),
        ("😠 Negatif", "Barang rusak saat tiba, sangat mengecewakan dan tidak sesuai."),
        ("😐 Netral",  "Barang sudah diterima. Pengiriman sesuai estimasi.")
    ]
    for col_btn, (lbl, ex) in zip([c1, c2, c3], examples):
        with col_btn:
            if st.button(lbl, use_container_width=True):
                st.session_state['main_input'] = ex
                st.rerun()

    analyze = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Hasil")
    if analyze and user_text.strip():
        with st.spinner("Menganalisis..."):
            sentiment, confidence, all_probs = predict_sentiment(user_text, model, tokenizer, config)

        s = sentiment.lower()
        if any(w in s for w in ['pos', 'positif', 'positive', 'baik', 'bagus']):
            css, label_display = 'positive', 'POSITIF'
        elif any(w in s for w in ['neg', 'negatif', 'negative', 'buruk']):
            css, label_display = 'negative', 'NEGATIF'
        else:
            css, label_display = 'neutral', 'NETRAL'

        st.markdown(
            f'<div class="result-card {css}">'
            f'{label_display}: {sentiment.upper()}<br>'
            f'<small>Confidence: {confidence*100:.1f}%</small>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("**Distribusi Probabilitas:**")
        for lbl, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            st.progress(float(prob), text=f"{lbl}: {prob*100:.1f}%")

        with st.expander("🔍 Detail Preprocessing"):
            st.write(f"**Teks asli:** {user_text[:200]}")
            st.write(f"**Teks bersih:** {clean_text(user_text)[:200]}")
            st.write(f"**Jumlah kata:** {len(clean_text(user_text).split())}")

    elif analyze:
        st.warning("Masukkan teks terlebih dahulu!")
    else:
        st.info("Masukkan teks di sebelah kiri lalu klik Analisis")

# ===================== BATCH =====================

st.markdown("---")
st.markdown("### 📋 Analisis Batch")
batch = st.text_area(
    "Beberapa teks, satu per baris:",
    placeholder="Teks pertama\nTeks kedua\nTeks ketiga",
    height=100
)

if st.button("📊 Analisis Semua"):
    if batch.strip():
        import pandas as pd
        texts = [t.strip() for t in batch.split('\n') if t.strip()]
        results = []
        bar = st.progress(0)
        for i, t in enumerate(texts):
            s, c, _ = predict_sentiment(t, model, tokenizer, config)
            results.append({
                'No': i + 1,
                'Teks': t[:60] + ('...' if len(t) > 60 else ''),
                'Sentimen': s,
                'Confidence': f"{c*100:.1f}%"
            })
            bar.progress((i + 1) / len(texts))
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)
        st.bar_chart(df_res['Sentimen'].value_counts())
    else:
        st.warning("Masukkan teks dulu!")

st.markdown("---")
st.caption("LSTM Sentiment Analysis | Text Mining | TensorFlow + Streamlit")
