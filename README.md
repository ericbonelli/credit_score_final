# 📑 Projeto de Credit Scoring — Regressão Logística

## 🔎 Visão Geral
Este projeto implementa e avalia um **modelo de Credit Scoring** utilizando **Regressão Logística**.  
O fluxo cobre desde o **pré-processamento dos dados** (missing, outliers, WOE/IV), avaliação com métricas (AUC, Gini, KS, PSI), até a **implementação de um app em Streamlit** para escoragem em lote de novos clientes.

---

## 📂 Estrutura do Repositório
```
.
├── notebooks/
│   ├── Mod37_Regressao_Logistica_II.ipynb   # exercícios de referência
│   ├── Mod38_Pipeline.ipynb                 # construção do pipeline
│   └── Mod38_PyCaret.ipynb                  # experimentos com PyCaret
├── model_final.pkl                          # modelo treinado e salvo
├── app_scoring_web.py                       # app Streamlit para escoragem
├── requirements.txt                         # dependências mínimas
├── exemplo_clientes.csv                     # exemplo sem target (apenas score)
├── exemplo_clientes_com_target.csv          # exemplo com target (para métricas)
└── README.md                                # este documento
```

---

## ⚙️ Preparação do Ambiente

### 1) Criar e ativar ambiente virtual (opcional)
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 2) Instalar dependências
```bash
pip install -r requirements.txt
```

Dependências principais:
- `scikit-learn`
- `pandas`, `numpy`, `scipy`
- `streamlit`
- `joblib`

---

## 📊 Desenvolvimento do Modelo

### Etapas
1. **Amostragem**
   - DEV: primeiras safras.
   - OOT: 3 últimas safras.
   - Exclusões: `data_ref`, `index`.

2. **Tratamento**
   - Zeros estruturais → missing/flags.
   - Outliers → winsorização 1% / 99%.
   - Categorias raras → “OUTROS”.
   - Renda → WOE supervisionado (6 bins).

3. **Modelagem**
   - Algoritmo: Regressão Logística.
   - Pipeline scikit-learn com imputação, OHE e logistic regression.
   - Modelo salvo em `model_final.pkl`.

4. **Avaliação**
   - AUC ~0.65, Gini ~0.30, KS ~0.21.
   - PSI Global ≈ 0.0 → modelo estável.
   - Decis mostram boa concentração: ~70% dos maus em 50% da carteira.

---

## 💻 Uso do Modelo Treinado

### Carregar o modelo em Python
```python
import joblib
import pandas as pd

# carregar
model = joblib.load("model_final.pkl")

# escorar um novo dataset (com mesmas colunas de treino)
df = pd.read_csv("novos_clientes.csv")
scores = model.predict_proba(df)[:, 1]
df["score"] = scores
```

---

## 🌐 App Streamlit — Escoragem em Lote

### Executar localmente
```bash
streamlit run app_scoring_web.py
```

### Funcionalidades
- Upload do **modelo (`.pkl`)** e do **CSV**.
- Alinhamento automático de colunas (exclui `data_ref`, `index`, etc.).
- Geração de coluna `score` (probabilidade de inadimplência).
- Se o CSV contiver `target`, exibe métricas:
  - AUC, Gini, KS, ACC.
  - Matriz de confusão.
  - Correção automática de probabilidade invertida.
- Download do CSV com scores.

### Deploy no Streamlit Cloud
1. Subir `app_scoring_web.py` + `requirements.txt` no GitHub.
2. Em [share.streamlit.io](https://share.streamlit.io), criar app apontando para o repositório.
3. Fazer upload do `model_final.pkl` via sidebar no app.

---

## 📄 Exemplos de Arquivos CSV

Para facilitar os testes, disponibilizamos dois arquivos de exemplo:

- [`exemplo_clientes.csv`](exemplo_clientes.csv)  
  ➝ contém apenas as variáveis de entrada necessárias para gerar os **scores**.  
  Útil para validar a escoragem em lote.

- [`exemplo_clientes_com_target.csv`](exemplo_clientes_com_target.csv)  
  ➝ contém as mesmas variáveis de entrada **+ a coluna `mau` (target)**.  
  Útil para calcular **métricas de avaliação** (AUC, Gini, KS, matriz de confusão) diretamente no app Streamlit.

### Estrutura esperada

| sexo | posse_de_veiculo | posse_de_imovel | qtd_filhos | tipo_renda  | educacao | estado_civil | tipo_residencia | idade | tempo_emprego | qt_pessoas_residencia | renda   | mau |
|------|------------------|-----------------|------------|-------------|----------|--------------|-----------------|-------|---------------|-----------------------|--------|-----|
| F    | N                | S               | 0          | Assalariado | Superior | Solteiro     | Casa            | 35    | 5             | 3                     | 3200.5 | 0   |
| M    | S                | N               | 2          | Empresário  | Médio    | Casado       | Apartamento     | 52    | 15            | 4                     | 12000  | 1   |
| F    | N                | N               | 1          | Autônomo    | Superior | Solteiro     | Casa            | 28    | 2             | 2                     | 2500   | 0   |

> ⚠️ Observação: a coluna `mau` é opcional.  
> - Se presente, o app Streamlit calculará automaticamente as métricas de performance.  
> - Se ausente, o app apenas adicionará a coluna `score` com a probabilidade de inadimplência.

---

## 📈 Conclusões e Recomendações
- **Modelo estável** (PSI ≈ 0).
- **Discriminância moderada** (AUC ~0.65).
- Renda WOE foi variável-chave.
- Recomendações:
  1. Monitorar PSI por safra.
  2. Reavaliar binning da renda periodicamente.
  3. Explorar WOE em idade e tempo de emprego.
  4. Testar gradient boosting como benchmark.
