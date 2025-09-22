
import io
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

st.set_page_config(page_title="Credit Scoring — Streamlit", layout="wide")
st.title("📊 Credit Scoring — Scoragem em Lote")

# =============================
# Sidebar — modelo e exclusões
# =============================
st.sidebar.header("⚙️ Configurações")

# Opção 1: usar arquivo default no app (ex.: quando rodar localmente)
MODEL_PATH_DEFAULT = "model_final.pkl"
use_uploaded_model = st.sidebar.checkbox("Enviar arquivo de modelo (.pkl)", value=True)

uploaded_model_file = None
model_obj = None
model_load_error = None

def load_model_bytes(b):
    # tenta joblib; se falhar, pickle
    try:
        return joblib.load(io.BytesIO(b))
    except Exception:
        return pickle.load(io.BytesIO(b))

if use_uploaded_model:
    uploaded_model_file = st.sidebar.file_uploader("Selecione o modelo (.pkl)", type=["pkl","joblib","pickle"])
    if uploaded_model_file is not None:
        try:
            model_bytes = uploaded_model_file.read()
            model_obj = load_model_bytes(model_bytes)
        except Exception as e:
            model_load_error = f"Falha ao carregar modelo enviado: {e}"
else:
    # usa arquivo no diretório do app
    if os.path.exists(MODEL_PATH_DEFAULT):
        try:
            with open(MODEL_PATH_DEFAULT, "rb") as f:
                try:
                    model_obj = joblib.load(f)
                except Exception:
                    f.seek(0)
                    model_obj = pickle.load(f)
        except Exception as e:
            model_load_error = f"Falha ao carregar {MODEL_PATH_DEFAULT}: {e}"
    else:
        model_load_error = f"Arquivo {MODEL_PATH_DEFAULT} não encontrado. Envie o modelo pela barra lateral."

# Colunas a excluir
excl_default = ['data_ref','_ref_date','index','mau','target']
excl_text = st.sidebar.text_area("Colunas para excluir (uma por linha):", "\n".join(excl_default))
EXCLUDE = set([x.strip() for x in excl_text.splitlines() if x.strip()])

# Mostrar status do modelo
if model_obj is not None:
    st.success(f"Modelo carregado ({type(model_obj).__name__}).")
else:
    st.error(model_load_error or "Modelo não carregado ainda.")
    st.stop()

# =============================
# Upload do CSV para score
# =============================
st.markdown("### 1) Carregue um CSV para score")
csv_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

def align_to_model_columns(df: pd.DataFrame, model):
    # tenta usar feature_names_in_ (sklearn >=1)
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        # tenta método get_feature_names_out (raros casos)
        try:
            if hasattr(model, "get_feature_names_out"):
                expected = list(model.get_feature_names_out())
        except Exception:
            pass
    if expected is None:
        # fallback: usa as colunas do arquivo menos EXCLUDE
        expected = [c for c in df.columns if c not in EXCLUDE]

    # adiciona colunas faltantes como NaN, na ordem esperada
    missing = [c for c in expected if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    df = df[expected].copy()
    return df, expected, missing

def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:,1].astype(float)
        return p.astype(float)
    if hasattr(model, "decision_function"):
        from scipy.special import expit
        return expit(model.decision_function(X)).astype(float)
    # fallback
    yhat = model.predict(X)
    return yhat.astype(float)

def youden_thr_safe(y, p):
    if len(np.unique(p)) == 1:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(thr[j])

def ensure_positive_proba(y, p):
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        return p, False
    if auc < 0.5:
        return 1 - p, True
    return p, False

if csv_file is not None:
    try:
        df_in = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()

    st.write("Prévia do arquivo:", df_in.head())

    # separa target (se existir) para avaliação
    y_true = None
    if "target" in df_in.columns:
        try:
            y_true = df_in["target"].astype(int).values
        except Exception:
            y_true = None

    # remove colunas explicitamente excluídas
    df_feat = df_in.drop(columns=[c for c in df_in.columns if c in EXCLUDE], errors="ignore")

    # alinhar
    X, expected, missing = align_to_model_columns(df_feat, model_obj)
    if missing:
        st.warning(f"Foram adicionadas {len(missing)} colunas ausentes como NaN para alinhar ao modelo.")

    # scoring
    try:
        proba = get_proba(model_obj, X)
    except Exception as e:
        st.error(f"Erro ao gerar score: {e}")
        st.stop()

    # avaliação (se houver target)
    metrics = {}
    if y_true is not None:
        p_corr, flipped = ensure_positive_proba(y_true, proba)
        thr = youden_thr_safe(y_true, p_corr)
        yhat = (p_corr >= thr).astype(int)
        auc = roc_auc_score(y_true, p_corr)
        fpr, tpr, _ = roc_curve(y_true, p_corr)
        ks = float(np.max(tpr - fpr))
        gini = 2*auc - 1
        acc = accuracy_score(y_true, yhat)
        cm = confusion_matrix(y_true, yhat)
        metrics = dict(flipped=flipped, thr=float(thr), auc=float(auc), gini=float(gini), ks=float(ks), acc=float(acc), cm=cm.tolist())

    # saída com score
    out = df_in.copy()
    out["score"] = proba

    st.markdown("### 2) Resultados")
    st.write("Prévia com 'score':", out.head())

    if metrics:
        st.markdown("#### Métricas (arquivo tinha 'target')")
        st.write({
            "Threshold (Youden)": round(metrics["thr"], 4),
            "AUC": round(metrics["auc"], 4),
            "Gini": round(metrics["gini"], 4),
            "KS": round(metrics["ks"], 4),
            "Acurácia": round(metrics["acc"], 4),
            "Prob invertida?": "Sim" if metrics["flipped"] else "Não"
        })
        st.write("Matriz de confusão:", np.array(metrics["cm"]))

    st.markdown("#### 3) Baixar arquivo com score")
    st.download_button(
        "⬇️ Baixar CSV com score",
        out.to_csv(index=False).encode("utf-8"),
        file_name="scored_output.csv",
        mime="text/csv"
    )

    with st.expander("🔎 Detalhes técnicos"):
        st.json({
            "modelo": type(model_obj).__name__,
            "n_linhas": int(len(df_in)),
            "colunas_entrada": list(df_in.columns),
            "colunas_usadas": expected,
            "colunas_excluidas": sorted(list(EXCLUDE)),
            "colunas_adicionadas_com_nan": missing,
            "tem_target": y_true is not None
        })
else:
    st.info("Faça upload de um CSV para iniciar a escoragem.")
