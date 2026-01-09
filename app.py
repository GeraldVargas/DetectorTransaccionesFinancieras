import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

st.set_page_config(
    page_title="Detección de Anomalías",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 4px;
    }
    .algo-indicator {
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
        text-align: center;
        font-weight: 500;
        background-color: #f0f8ff;
        border: 1px solid #4a90e2;
        color: #2c5282;
    }
    .algo-name {
        font-size: 15px;
        font-weight: 600;
        color: #2c5282;
    }
    .algo-desc {
        font-size: 12px;
        color: #4a5568;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

CONFIG = {
    "datasets": {
        "Credit Card Fraud": {
            "id": "creditcard",
            "features_base": ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                              'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                              'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'],
            "drop_cols": ["Class", "Time"]
        },
        "PaySim Financial": {
            "id": "paysim",
            "features_base": ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type'],
            "drop_cols": ["isFraud", "isFlaggedFraud", "step", "nameOrig", "nameDest"]
        }
    },
    "algoritmos": {
        "Isolation Forest": {
            "id": "if",
            "desc": "Aísla anomalías usando árboles de decisión"
        },
        "Autoencoder": {
            "id": "ae",
            "desc": "Red neuronal que aprende reconstrucción de datos"
        }
    }
}

@st.cache_resource
def load_audit_engine(dataset_id, algo_id):
    resources = {}
    model_dir = "models"
    
    try:
        if algo_id == "if":
            path = f"{model_dir}/if_{dataset_id}.joblib"
            if os.path.exists(path):
                resources['model'] = joblib.load(path)
                resources['type'] = 'statistical'
                resources['algo_name'] = 'Isolation Forest'
                resources['algo_id'] = 'if'
            else:
                st.warning(f"Modelo Isolation Forest para {dataset_id} no encontrado")
                return None
                
        elif algo_id == "ae":
            model_path = f"{model_dir}/autoencoder_{dataset_id}.keras"
            scaler_path = f"{model_dir}/scaler_{dataset_id}.joblib"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                resources['model'] = load_model(model_path)
                resources['scaler'] = joblib.load(scaler_path)
                resources['type'] = 'neural'
                resources['algo_name'] = 'Autoencoder'
                resources['algo_id'] = 'ae'
            else:
                st.warning(f"Modelo Autoencoder para {dataset_id} no encontrado")
                return None
                
        return resources
        
    except Exception as e:
        st.error(f"Error cargando modelo: {str(e)}")
        return None

def standardize_input(df, dataset_conf):
    X = df.copy()
    ds_id = dataset_conf["id"]
    
    if ds_id == "paysim":
        if 'type' in X.columns:
            if X['type'].dtype == 'object':
                tipo_map = {
                    'TRANSFER': 0,
                    'CASH_OUT': 1,
                    'PAYMENT': 0,
                    'DEBIT': 0,
                    'CASH_IN': 0
                }
                X['type'] = X['type'].map(tipo_map).fillna(0)
        
        expected = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                   'oldbalanceDest', 'newbalanceDest']
        
        for c in expected:
            if c not in X.columns: 
                X[c] = 0.0
      
        X = X[expected]
        
    elif ds_id == "creditcard":
        for col in dataset_conf["drop_cols"]:
            if col in X.columns: 
                X = X.drop(col, axis=1)
                
        expected = dataset_conf["features_base"]
        for c in expected:
            if c not in X.columns: 
                X[c] = 0.0
        X = X[expected]
        
    return X.fillna(0)

def create_risk_pie_chart(results, algo_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    counts = results['Estado'].value_counts()
    
    colors = []
    labels = []
    sizes = []
    
    if 'FRAUDE' in counts.index:
        sizes.append(counts['FRAUDE'])
        labels.append('FRAUDE')
        colors.append('#e74c3c')
    
    if 'NORMAL' in counts.index:
        sizes.append(counts['NORMAL'])
        labels.append('NORMAL')
        colors.append('#2ecc71')
    
    if not sizes:
        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', fontsize=12)
        ax.set_title(f'Distribución - {algo_name}', fontsize=14, fontweight='bold')
        return fig
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title(f'Distribución - {algo_name}', fontsize=14, fontweight='bold')
    
    return fig

def create_risk_score_chart(scores, results, algo_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if len(scores) == 0:
        ax.text(0.5, 0.5, 'Sin datos para graficar', ha='center', va='center', fontsize=12)
        ax.set_title(f'Scores de Riesgo - {algo_name}', fontsize=14, fontweight='bold')
        return fig
    
    normal_mask = results['Estado'] == 'NORMAL'
    fraude_mask = results['Estado'] == 'FRAUDE'
    
    normal_scores = scores[normal_mask]
    fraude_scores = scores[fraude_mask]
    
    bins = min(30, max(10, len(scores)//10))
    
    if len(normal_scores) > 0:
        ax.hist(normal_scores, bins=bins, alpha=0.7, color='#2ecc71', label='NORMAL', 
                density=True, edgecolor='black', linewidth=0.5)
    
    if len(fraude_scores) > 0:
        ax.hist(fraude_scores, bins=bins, alpha=0.7, color='#e74c3c', label='FRAUDE',
                density=True, edgecolor='black', linewidth=0.5)
    
    if len(scores) > 0:
        mean_score = np.mean(scores)
        ax.axvline(x=mean_score, color='#3498db', linestyle='--', linewidth=2, 
                   label=f'Media: {mean_score:.3f}')
    
    ax.set_xlabel('Score de Riesgo', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title(f'Scores de Riesgo - {algo_name}', fontsize=14, fontweight='bold')
    
    handles, labels_legend = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=11)
    
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

with st.sidebar:
    st.title("Configuración")
    
    dataset_key = st.selectbox("Dataset", list(CONFIG["datasets"].keys()))
    dataset_conf = CONFIG["datasets"][dataset_key]
    
    algo_key = st.selectbox("Algoritmo", list(CONFIG["algoritmos"].keys()))
    algo_info = CONFIG["algoritmos"][algo_key]
    algo_id = algo_info["id"]
    
    st.divider()
    
    st.subheader("Algoritmo Activo")
    
    st.markdown(
        f'<div class="algo-indicator">'
        f'<div class="algo-name">{algo_key.upper()}</div>'
        f'<div class="algo-desc">{algo_info["desc"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    if 'resources' in st.session_state and st.session_state['resources']:
        res = st.session_state['resources']
        st.caption(f"Modelo cargado: {res['algo_name']}")
    
    if st.button("Recargar Modelo", key="reload_model"):
        if 'resources' in st.session_state:
            del st.session_state['resources']
        st.rerun()

st.title("Sistema de Detección de Anomalías Financieras")

tab_manual, tab_batch = st.tabs(["Evaluación Individual", "Evaluación por Lotes"])

with tab_manual:
    st.subheader("Transacción Individual")
    input_dict = {}
    
    if dataset_conf["id"] == "paysim":
        c1, c2, c3 = st.columns(3)
        
        with c1:
            input_dict['type'] = st.selectbox("Tipo", 
                ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
            input_dict['amount'] = st.number_input("Monto", min_value=0.0, 
                                                   value=1000.0, step=100.0)
        
        with c2:
            input_dict['oldbalanceOrg'] = st.number_input("Saldo Origen (Antes)", 
                                                          min_value=0.0)
            input_dict['newbalanceOrig'] = st.number_input("Saldo Origen (Después)", 
                                                           min_value=0.0)
        
        with c3:
            input_dict['oldbalanceDest'] = st.number_input("Saldo Destino (Antes)", 
                                                           min_value=0.0)
            input_dict['newbalanceDest'] = st.number_input("Saldo Destino (Después)", 
                                                           min_value=0.0)

    elif dataset_conf["id"] == "creditcard":
        col_main, col_grid = st.columns([1, 3])
        
        with col_main:
            input_dict['Amount'] = st.number_input("Monto", min_value=0.0, 
                                                   value=150.0)
        
        with col_grid:
            with st.expander("V1 - V28", expanded=True):
                grid_cols = st.columns(4)
                for i in range(1, 29):
                    col_idx = (i-1) % 4
                    with grid_cols[col_idx]:
                        input_dict[f'V{i}'] = st.number_input(f"V{i}", value=0.0, 
                                                             step=0.1, key=f"v{i}")

    if st.button("Evaluar Transacción", use_container_width=True, key="eval_manual"):
        df_manual = pd.DataFrame([input_dict])
        
        st.session_state['data_to_process'] = df_manual
        st.session_state['mode'] = 'manual'
        st.session_state['current_algo'] = algo_id
        st.session_state['current_dataset'] = dataset_conf["id"]
        
        st.rerun()

with tab_batch:
    st.subheader("Carga de Archivo CSV")
    
    uploaded_file = st.file_uploader("Seleccionar archivo", type=["csv"], 
                                     key="file_uploader")
    
    if uploaded_file is not None:
        if uploaded_file.size == 0:
            st.error("ERROR: El archivo está vacío")
        else:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.success(f"Archivo cargado: {df_uploaded.shape[0]} registros")
                
                if st.button("Procesar Lote", use_container_width=True, 
                           key="process_batch"):
                    st.session_state['data_to_process'] = df_uploaded
                    st.session_state['mode'] = 'batch'
                    st.session_state['current_algo'] = algo_id
                    st.session_state['current_dataset'] = dataset_conf["id"]
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if 'data_to_process' in st.session_state:
    st.divider()
    
    algo_changed = ('current_algo' in st.session_state and 
                   st.session_state['current_algo'] != algo_id)
    
    dataset_changed = ('current_dataset' in st.session_state and 
                      st.session_state['current_dataset'] != dataset_conf["id"])
    
    if algo_changed or dataset_changed:
        st.warning("⚠️ La configuración cambió. Por favor, procesa los datos nuevamente.")
        if st.button("Procesar con nueva configuración", key="reprocess"):
            st.session_state['current_algo'] = algo_id
            st.session_state['current_dataset'] = dataset_conf["id"]
            st.rerun()
    
    else:
        st.subheader("Resultados del Análisis")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"**Dataset:** {dataset_key}")
        with col_info2:
            st.info(f"**Algoritmo:** {algo_key}")
        
        df_raw = st.session_state['data_to_process']
        
        resources = load_audit_engine(dataset_conf["id"], algo_id)

        if not resources:
            st.error(f"No se encontró el modelo {algo_key} para {dataset_key}")
            st.info("""
            **Posibles soluciones:**
            1. Entrena los modelos primero (ejecuta los notebooks)
            2. Verifica que los archivos .joblib/.keras estén en la carpeta 'models/'
            3. Los nombres deben ser: if_creditcard.joblib, autoencoder_creditcard.keras, etc.
            """)
        
        else:
            try:
                with st.spinner(f"Procesando con {resources['algo_name']}..."):
                    
                    X = standardize_input(df_raw, dataset_conf)
                    
                    if resources['type'] == 'statistical':
                        model = resources['model']
                        
                        scores = model.decision_function(X)
                        
                        preds = model.predict(X)
                        
                        results = df_raw.copy()
                        results['Risk_Score'] = scores
                        results['Prediction'] = preds
                        results['Estado'] = results['Prediction'].apply(
                            lambda x: 'FRAUDE' if x == -1 else 'NORMAL'
                        )
                        
                    elif resources['type'] == 'neural':
                        model = resources['model']
                        scaler = resources['scaler']
                        
                        X_scaled = scaler.transform(X)
                        
                        reconst = model.predict(X_scaled, verbose=0)
                        
                        mse = np.mean(np.power(X_scaled - reconst, 2), axis=1)
                        
                        scores = -mse
                        
                        if len(mse) == 1:
                            thresh = 0.05 if dataset_conf["id"] == "creditcard" else 0.1
                            preds = [-1 if mse[0] > thresh else 1]
                        else:
                            thresh = np.percentile(mse, 98)
                            preds = [-1 if e > thresh else 1 for e in mse]
                        
                        results = df_raw.copy()
                        results['Risk_Score'] = scores
                        results['Prediction'] = preds
                        results['Estado'] = results['Prediction'].apply(
                            lambda x: 'FRAUDE' if x == -1 else 'NORMAL'
                        )
                    
                    total = len(results)
                    anomalias = sum(results['Estado'] == 'FRAUDE')
                    normales = total - anomalias
                    tasa_riesgo = (anomalias / total * 100) if total > 0 else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Procesado", f"{total:,}")
                    with col2:
                        st.metric("Transacciones Normales", f"{normales:,}")
                    with col3:
                        st.metric("Anomalías Detectadas", f"{anomalias:,}")
                    with col4:
                        st.metric("Tasa de Riesgo", f"{tasa_riesgo:.2f}%")
                    
                    st.caption(f"**Algoritmo utilizado:** {resources['algo_name']} | "
                              f"**Score promedio:** {np.mean(scores):.4f}")
                    
                    st.divider()
                    
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        fig1 = create_risk_pie_chart(results, resources['algo_name'])
                        st.pyplot(fig1)
                        plt.close(fig1)
                    
                    with col_chart2:
                        fig2 = create_risk_score_chart(scores, results, resources['algo_name'])
                        st.pyplot(fig2)
                        plt.close(fig2)
                    
                    st.divider()
                    
                    st.subheader("Transacciones Analizadas")
                    
                    fraudes_detectados = results[results['Estado'] == 'FRAUDE'].copy()
                    
                    def color_estado(val):
                        if val == 'FRAUDE':
                            return 'background-color: #ffcccc; color: #721c24; font-weight: bold'
                        else:
                            return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    
                    if len(fraudes_detectados) > 0:
                        styled_fraudes = fraudes_detectados.style.applymap(color_estado, subset=['Estado'])
                        
                        st.write(f"**Se detectaron {len(fraudes_detectados)} transacciones fraudulentas:**")
                        st.dataframe(
                            styled_fraudes,
                            use_container_width=True,
                            height=min(400, len(fraudes_detectados) * 35 + 100)
                        )
                        
                        with st.expander("📊 Estadísticas de las transacciones fraudulentas"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if 'Amount' in fraudes_detectados.columns:
                                    st.metric("Monto promedio", f"${fraudes_detectados['Amount'].mean():.2f}")
                            with col2:
                                if 'Risk_Score' in fraudes_detectados.columns:
                                    st.metric("Score de riesgo promedio", f"{fraudes_detectados['Risk_Score'].mean():.3f}")
                            with col3:
                                if 'Amount' in fraudes_detectados.columns:
                                    st.metric("Monto total detectado", f"${fraudes_detectados['Amount'].sum():.2f}")
                    
                    else:
                        st.success("✅ No se detectaron transacciones fraudulentas")
                        st.dataframe(
                            pd.DataFrame({"Mensaje": ["No hay transacciones fraudulentas para mostrar"]}),
                            use_container_width=True,
                            height=100
                        )
                    
                    if len(fraudes_detectados) > 0:
                        csv = fraudes_detectados.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label=f"📥 Descargar {len(fraudes_detectados)} fraudes detectados",
                            data=csv,
                            file_name=f"fraudes_detectados_{dataset_conf['id']}_{algo_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help="Descarga solo las transacciones marcadas como fraudulentas"
                        )
                    else:
                        st.download_button(
                            label="📥 Descargar fraudes detectados",
                            data="",  
                            file_name="",
                            mime="text/csv",
                            use_container_width=True,
                            disabled=True,
                            help="No hay fraudes para descargar"
                        )
                    
                    st.divider()
                    
                    if st.button("Realizar nuevo análisis", use_container_width=True, key="new_analysis"):
                        if 'data_to_process' in st.session_state:
                            del st.session_state['data_to_process']
                        if 'current_algo' in st.session_state:
                            del st.session_state['current_algo']
                        if 'current_dataset' in st.session_state:
                            del st.session_state['current_dataset']
                        st.rerun()
            
            except Exception as e:
                st.error(f"Error en el procesamiento: {str(e)}")
                st.info("""
                **Posibles causas:**
                1. El CSV no tiene las columnas esperadas
                2. Los datos no están en el formato correcto
                3. Problema con el modelo entrenado
                """)

