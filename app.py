import streamlit as st
import numpy as np
import pickle
import json
import re
import os
import torch
import torch.nn as nn


class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out)


@st.cache_resource
def load_all():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, 'model_config.json')) as f:
        config = json.load(f)
    with open(os.path.join(base, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    model = BiLSTMSentiment(config['VOCAB_SIZE'], config['EMBED_DIM'], config['HIDDEN_DIM'], config['num_classes'])
    model.load_state_dict(torch.load(os.path.join(base, 'lstm_model.pt'), map_location='cpu'))
    model.eval()
    return model, word2idx, config


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict(text, model, word2idx, config):
    cleaned = clean_text(text)
    tokens = cleaned.split()[:config['MAX_LEN']]
    ids = [word2idx.get(t, 1) for t in tokens]
    ids += [0] * (config['MAX_LEN'] - len(ids))
    x = torch.LongTensor([ids])
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).numpy()[0]
    pred_idx = int(probs.argmax())
    label_classes = config['label_classes']
    all_probs = {str(label_classes[i]): float(probs[i]) for i in range(len(label_classes))}
    return str(label_classes[pred_idx]), float(probs[pred_idx]), all_probs


st.set_page_config(page_title='Sentiment Analysis LSTM', page_icon='🧠', layout='wide')
st.markdown('<style>.result-card{padding:1.5rem;border-radius:12px;text-align:center;font-size:1.2rem;font-weight:600;margin:1rem 0}.positive{background:#d1fae5;color:#065f46;border:2px solid #34d399}.negative{background:#fee2e2;color:#991b1b;border:2px solid #f87171}.neutral{background:#e0f2fe;color:#0369a1;border:2px solid #38bdf8}</style>', unsafe_allow_html=True)
st.title('Sentiment Analysis - Bidirectional LSTM')
st.caption('Text Mining | PyTorch')

with st.spinner('Loading model...'):
    model, word2idx, config = load_all()

with st.sidebar:
    st.markdown('### Model Info')
    st.metric('Test Accuracy', f"{config['test_accuracy']*100:.1f}%")
    st.metric('F1 Score', f"{config['f1_score']:.4f}")
    st.markdown('---')
    st.write(f"Vocab size: {config['VOCAB_SIZE']:,}")
    st.write(f"Max length: {config['MAX_LEN']} tokens")
    st.write(f"Kelas: {', '.join([str(c) for c in config['label_classes']])}")
    base = os.path.dirname(os.path.abspath(__file__))
    for fname, title in [('training_history.png','Training History'),('confusion_matrix.png','Confusion Matrix')]:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            st.markdown(f'### {title}')
            st.image(p)

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown('### Masukkan Teks')
    user_text = st.text_area('Teks:', placeholder='Ketik teks di sini...', height=150, key='main_input')
    c1, c2, c3 = st.columns(3)
    examples = [('Positif','Produk sangat memuaskan, kualitas terbaik!'),('Negatif','Barang rusak saat tiba, sangat mengecewakan.'),('Netral','Barang sudah diterima sesuai estimasi.')]
    for col_btn, (lbl, ex) in zip([c1,c2,c3], examples):
        with col_btn:
            if st.button(lbl, use_container_width=True):
                st.session_state['main_input'] = ex
                st.rerun()
    analyze = st.button('Analisis Sentimen', type='primary', use_container_width=True)

with col2:
    st.markdown('### Hasil')
    if analyze and user_text.strip():
        with st.spinner('Menganalisis...'):
            sentiment, confidence, all_probs = predict(user_text, model, word2idx, config)
        s = sentiment.lower()
        css = 'positive' if any(w in s for w in ['pos','positif','positive','baik','bagus']) else 'negative' if any(w in s for w in ['neg','negatif','negative','buruk']) else 'neutral'
        st.markdown(f'<div class="result-card {css}"><b>{sentiment.upper()}</b><br><small>Confidence: {confidence*100:.1f}%</small></div>', unsafe_allow_html=True)
        st.markdown('**Distribusi Probabilitas:**')
        for lbl, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            st.progress(float(prob), text=f'{lbl}: {prob*100:.1f}%')
        with st.expander('Detail'):
            st.write(f'Asli: {user_text[:200]}')
            st.write(f'Bersih: {clean_text(user_text)[:200]}')
    elif analyze:
        st.warning('Masukkan teks dulu!')
    else:
        st.info('Masukkan teks lalu klik Analisis')

st.markdown('---')
st.markdown('### Analisis Batch')
batch = st.text_area('Beberapa teks (satu per baris):', height=100)
if st.button('Analisis Semua'):
    if batch.strip():
        import pandas as pd
        texts = [t.strip() for t in batch.split('\n') if t.strip()]
        results, bar = [], st.progress(0)
        for i, t in enumerate(texts):
            s, c, _ = predict(t, model, word2idx, config)
            results.append({'No':i+1,'Teks':t[:60],'Sentimen':s,'Confidence':f'{c*100:.1f}%'})
            bar.progress((i+1)/len(texts))
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.bar_chart(pd.DataFrame(results)['Sentimen'].value_counts())

st.markdown('---')
st.caption('LSTM Sentiment Analysis | Text Mining | PyTorch + Streamlit')
