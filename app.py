import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="APP Mega-Sena",
    page_icon="🎲",
    layout="wide"
)

# Configurar estilo dos gráficos
# Tenta usar o estilo específico, caso não exista na versão instalada, usa um padrão
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')

sns.set_palette("husl")

# Título principal
st.title("🎲 APP de Análise da Mega-Sena")
st.markdown("---")

# Upload do arquivo
st.write('Baixe a planilha através do site: https://www.lotocerta.com.br/todos-os-resultados-mega-sena-em-planilha-excel/ >> (opção de "Gerar Planilha de Resultados)') 
uploaded_file = st.file_uploader(
    "📁 Arraste ou selecione a planilha Excel com os resultados da Mega-Sena",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    # Opções de análise (Layout em colunas, sem sidebar)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        opcao = st.radio(
            "📊 Escolha o período de análise:",
            ["Todos os sorteios", "Últimos N sorteios", "Primeiros N sorteios"],
            horizontal=True
        )
    
    with col2:
        n_sorteios = 500 # Valor padrão
        if opcao != "Todos os sorteios":
            n_sorteios = st.number_input(
                "Quantidade de sorteios:",
                min_value=10,
                max_value=10000,
                value=500,
                step=50
            )
    
    if st.button("🚀 Gerar Análises", type="primary", use_container_width=True):
        with st.spinner("Processando dados e gerando análises..."):
            try:
                # Carregar dados
                # O pandas detecta automaticamente o engine, mas o openpyxl precisa estar instalado
                df_raw = pd.read_excel(uploaded_file)
                
                # Filtragem dos dados baseada na escolha
                if opcao == "Todos os sorteios":
                    df = df_raw.copy()
                    periodo_analise = "TODOS os sorteios"
                elif opcao == "Últimos N sorteios":
                    df = df_raw.tail(n_sorteios).reset_index(drop=True)
                    periodo_analise = f"ÚLTIMOS {n_sorteios} sorteios"
                else:
                    df = df_raw.head(n_sorteios).reset_index(drop=True)
                    periodo_analise = f"PRIMEIROS {n_sorteios} sorteios"
                
                # Seleção das colunas de bolas (Assumindo colunas C até H / índices 2 a 7)
                # Ajuste o iloc se sua planilha tiver um formato diferente
                bolas_df = df.iloc[:, 2:8]
                
                # Informações gerais
                st.success(f"✅ Análise concluída! Processados {len(df)} sorteios")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Período", periodo_analise)
                with col2:
                    st.metric("🎯 Total de Sorteios", len(df))
                with col3:
                    concurso_final = df.iloc[-1, 0] if df.shape[1] > 0 else 'N/A'
                    st.metric("🏁 Último Concurso", str(concurso_final))
                
                st.markdown("---")
                
                # ============================================
                # 1. FREQUÊNCIA GERAL
                # ============================================
                st.header("1️ - Frequência Geral dos Números")
                
                s = bolas_df.stack()
                frequencia = s.value_counts().sort_index()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔥 Mais Sorteado", f"{frequencia.idxmax()} ({frequencia.max()}x)")
                with col2:
                    st.metric("❄️ Menos Sorteado", f"{frequencia.idxmin()} ({frequencia.min()}x)")
                with col3:
                    st.metric("📊 Média", f"{frequencia.mean():.1f}")
                with col4:
                    st.metric("📈 Desvio Padrão", f"{frequencia.std():.1f}")
                
                fig, ax = plt.subplots(figsize=(18, 6))
                colors = ['red' if x == frequencia.max() else 'green' if x == frequencia.min() else 'steelblue' 
                          for x in frequencia.values]
                ax.bar(frequencia.index, frequencia.values, color=colors, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Número', fontsize=12, fontweight='bold')
                ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
                ax.set_title('Frequência de Cada Número Sorteado', fontsize=14, fontweight='bold')
                ax.set_xticks(range(1, 61))
                ax.axhline(frequencia.mean(), color='orange', linestyle='--', linewidth=2, 
                          label=f'Média: {frequencia.mean():.1f}')
                ax.legend(fontsize=10)
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                with st.expander("📋 Ver detalhes - Top 10 e Bottom 10"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🔝 Top 10 Mais Sorteados")
                        for i, (num, freq) in enumerate(frequencia.nlargest(10).items(), 1):
                            st.write(f"{i}. Número **{num}**: {freq} vezes")
                    with col2:
                        st.subheader("🔻 Top 10 Menos Sorteados")
                        for i, (num, freq) in enumerate(frequencia.nsmallest(10).items(), 1):
                            st.write(f"{i}. Número **{num}**: {freq} vezes")
                
                st.markdown("---")
                
                # ============================================
                # 2. PARES FREQUENTES
                # ============================================
                st.header("2️ - Pares que Mais Saíram Juntos")
                
                pares = []
                for _, row in bolas_df.iterrows():
                    numeros = sorted(row.values)
                    for i in range(len(numeros)):
                        for j in range(i+1, len(numeros)):
                            pares.append(tuple(sorted([int(numeros[i]), int(numeros[j])])))
                
                pares_freq = Counter(pares)
                top_pares = pares_freq.most_common(15)
                
                fig, ax = plt.subplots(figsize=(14, 8))
                pares_labels = [f"{p[0]:02d}-{p[1]:02d}" for p, _ in top_pares]
                pares_valores = [f for _, f in top_pares]
                ax.barh(pares_labels, pares_valores, color='teal', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Frequência', fontsize=12, fontweight='bold')
                ax.set_ylabel('Par de Números', fontsize=12, fontweight='bold')
                ax.set_title('15 Pares Mais Frequentes', fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                with st.expander("📋 Ver lista completa dos 15 pares"):
                    for i, (par, freq) in enumerate(top_pares, 1):
                        st.write(f"{i}. **{par[0]:02d}-{par[1]:02d}**: {freq} vezes")
                
                st.markdown("---")
                
                # ============================================
                # 3. TRIOS FREQUENTES
                # ============================================
                st.header("3️ - Trios que Mais Saíram Juntos")
                
                trios = []
                for _, row in bolas_df.iterrows():
                    numeros = sorted(row.values)
                    for i in range(len(numeros)):
                        for j in range(i+1, len(numeros)):
                            for k in range(j+1, len(numeros)):
                                trios.append(tuple(sorted([int(numeros[i]), int(numeros[j]), int(numeros[k])])))
                
                trios_freq = Counter(trios)
                top_trios = trios_freq.most_common(10)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("🥇 Top 10 Trios")
                    for i, (trio, freq) in enumerate(top_trios, 1):
                        st.write(f"{i}. **{trio[0]:02d}-{trio[1]:02d}-{trio[2]:02d}**: {freq} vezes")
                
                with col2:
                    trios_labels = [f"{t[0]:02d}-{t[1]:02d}-{t[2]:02d}" for t, _ in top_trios]
                    trios_valores = [f for _, f in top_trios]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.barh(trios_labels, trios_valores, color='purple', alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Frequência', fontsize=10, fontweight='bold')
                    ax.set_title('Top 10 Trios', fontsize=12, fontweight='bold')
                    ax.invert_yaxis()
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.markdown("---")
                
                
                # ============================================
                # 4. NÚMEROS ATRASADOS
                # ============================================
                st.header("4️ -  Números 'Atrasados'")
                
                ultima_aparicao = {}
                for idx, row in df.iterrows():
                    concurso = idx # Usando índice como referência temporal relativa
                    for num in row.iloc[2:8]:
                        ultima_aparicao[int(num)] = concurso
                
                numeros_atrasados = []
                ultimo_concurso = len(df) - 1
                for num in range(1, 61):
                    if num in ultima_aparicao:
                        atraso = ultimo_concurso - ultima_aparicao[num]
                        numeros_atrasados.append((num, atraso))
                    else:
                        numeros_atrasados.append((num, ultimo_concurso))
                
                numeros_atrasados.sort(key=lambda x: x[1], reverse=True)
                
                fig, ax = plt.subplots(figsize=(16, 6))
                nums = [n for n, _ in numeros_atrasados]
                atrasos = [a for _, a in numeros_atrasados]
                colors_atraso = ['red' if a > 100 else 'orange' if a > 50 else 'green' for a in atrasos]
                ax.bar(nums, atrasos, color=colors_atraso, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Número', fontsize=12, fontweight='bold')
                ax.set_ylabel('Sorteios sem Aparecer', fontsize=12, fontweight='bold')
                ax.set_title('Números "Atrasados" - Quantidade de Sorteios desde a Última Aparição', 
                            fontsize=14, fontweight='bold')
                ax.set_xticks(range(1, 61))
                ax.axhline(50, color='orange', linestyle='--', alpha=0.5, label='50 sorteios')
                ax.axhline(100, color='red', linestyle='--', alpha=0.5, label='100 sorteios')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                with st.expander("📋 Ver top 15 números mais atrasados"):
                    for i, (num, atraso) in enumerate(numeros_atrasados[:15], 1):
                        st.write(f"{i}. Número **{num:02d}**: {atraso} sorteios sem aparecer")
                
                st.markdown("---")
                
            
                
                # ============================================
                # 5. ANÁLISE TEMPORAL
                # ============================================
                st.header("5 - Análise Temporal")
                
                total_sorteios = len(df)
                periodo1 = total_sorteios // 3
                periodo2 = 2 * periodo1
                
                freq_p1 = df.iloc[:periodo1, 2:8].stack().value_counts()
                freq_p2 = df.iloc[periodo1:periodo2, 2:8].stack().value_counts()
                freq_p3 = df.iloc[periodo2:, 2:8].stack().value_counts()
                
                st.info(f"""
                📊 **Divisão dos períodos:**
                - Período 1: sorteios 0 a {periodo1-1} ({periodo1} sorteios)
                - Período 2: sorteios {periodo1} a {periodo2-1} ({periodo2-periodo1} sorteios)
                - Período 3: sorteios {periodo2} a {total_sorteios-1} ({total_sorteios-periodo2} sorteios)
                """)
                
                top_15_geral = frequencia.nlargest(15).index
                df_temporal = pd.DataFrame({
                    'Período 1': freq_p1,
                    'Período 2': freq_p2,
                    'Período 3': freq_p3
                }).fillna(0).astype(int)
                
                fig, ax = plt.subplots(figsize=(16, 8))
                x = np.arange(len(top_15_geral))
                width = 0.25
                
                bars1 = ax.bar(x - width, [df_temporal.loc[n, 'Período 1'] if n in df_temporal.index else 0 for n in top_15_geral], 
                              width, label='Período 1', alpha=0.8)
                bars2 = ax.bar(x, [df_temporal.loc[n, 'Período 2'] if n in df_temporal.index else 0 for n in top_15_geral], 
                              width, label='Período 2', alpha=0.8)
                bars3 = ax.bar(x + width, [df_temporal.loc[n, 'Período 3'] if n in df_temporal.index else 0 for n in top_15_geral], 
                              width, label='Período 3', alpha=0.8)
                
                ax.set_xlabel('Número', fontsize=12, fontweight='bold')
                ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
                ax.set_title('Evolução Temporal dos 15 Números Mais Sorteados', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(top_15_geral)
                ax.legend(fontsize=11)
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                with st.expander("📋 Ver top 5 de cada período"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.subheader("Período 1")
                        for i, (num, freq) in enumerate(freq_p1.head(5).items(), 1):
                            st.write(f"{i}. Número **{num}**: {freq}x")
                    with col2:
                        st.subheader("Período 2")
                        for i, (num, freq) in enumerate(freq_p2.head(5).items(), 1):
                            st.write(f"{i}. Número **{num}**: {freq}x")
                    with col3:
                        st.subheader("Período 3")
                        for i, (num, freq) in enumerate(freq_p3.head(5).items(), 1):
                            st.write(f"{i}. Número **{num}**: {freq}x")
                
                st.markdown("---")
                
                # ============================================
                # 6. HEATMAP DE CO-OCORRÊNCIA
                # ============================================
                st.header("6 - Heatmap de Co-ocorrência")
                
                coocorrencia = np.zeros((60, 60))
                
                for _, row in bolas_df.iterrows():
                    numeros = [int(x) for x in row.values]
                    for i in numeros:
                        for j in numeros:
                            if i != j:
                                coocorrencia[i-1][j-1] += 1
                
                top_20 = frequencia.nlargest(20).index.tolist()
                indices = [n-1 for n in top_20]
                cooc_sample = coocorrencia[np.ix_(indices, indices)]
                
                fig, ax = plt.subplots(figsize=(14, 12))
                sns.heatmap(cooc_sample, 
                           xticklabels=top_20, 
                           yticklabels=top_20,
                           cmap='YlOrRd',
                           annot=True,
                           fmt='.0f',
                           cbar_kws={'label': 'Frequência de Co-ocorrência'},
                           linewidths=0.5,
                           ax=ax)
                ax.set_title('Heatmap de Co-ocorrência - Top 20 Números Mais Sorteados', 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('Número', fontsize=12, fontweight='bold')
                ax.set_ylabel('Número', fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
                pares_cooc = []
                for i in range(60):
                    for j in range(i+1, 60):
                        pares_cooc.append(((i+1, j+1), coocorrencia[i][j]))
                
                pares_cooc.sort(key=lambda x: x[1], reverse=True)
                
                with st.expander("📋 Ver top 10 pares com maior co-ocorrência"):
                    for i, (par, freq) in enumerate(pares_cooc[:10], 1):
                        st.write(f"{i}. **{par[0]:02d}-{par[1]:02d}**: {int(freq)} vezes")
                
                # ============================================
                # AVISO FINAL (Recuperado)
                # ============================================
                st.markdown("---")
                st.warning("""
                ⚠️ **LEMBRETE IMPORTANTE:**
                
                Todas essas análises são puramente estatísticas e descritivas.
                A Mega-Sena é um jogo de azar onde cada sorteio é independente.
                
                **Padrões históricos NÃO aumentam as chances de prever resultados futuros.**
                """)

            except Exception as e:
                st.error(f"Erro ao processar a planilha. Verifique se o arquivo está no formato correto da Mega-Sena (Caixa). Detalhe do erro: {e}")