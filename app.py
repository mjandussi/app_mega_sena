import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from itertools import combinations # Importante para a análise combinada

# Configuração da página
st.set_page_config(
    page_title="APP Mega-Sena",
    page_icon="🎲",
    layout="wide"
)

# Configurar estilo dos gráficos
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
    # Opções de análise
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
    
    # --- CORREÇÃO AQUI ---
    # Inicializa o estado se não existir
    if 'analise_gerada' not in st.session_state:
        st.session_state['analise_gerada'] = False

    # Botão apenas atualiza o estado
    if st.button("🚀 Gerar Análises", type="primary", use_container_width=True):
        st.session_state['analise_gerada'] = True

    # O código agora verifica o ESTADO, não o botão diretamente
    if st.session_state['analise_gerada']:
        with st.spinner("Processando dados e gerando análises..."):
            try:
                # Carregar dados
                df_raw = pd.read_excel(uploaded_file)
                
                # Filtragem dos dados
                if opcao == "Todos os sorteios":
                    df = df_raw.copy()
                    periodo_analise = "TODOS os sorteios"
                elif opcao == "Últimos N sorteios":
                    df = df_raw.tail(n_sorteios).reset_index(drop=True)
                    periodo_analise = f"ÚLTIMOS {n_sorteios} sorteios"
                else:
                    df = df_raw.head(n_sorteios).reset_index(drop=True)
                    periodo_analise = f"PRIMEIROS {n_sorteios} sorteios"
                
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
                # NOVA SEÇÃO: ANÁLISE DE NÚMEROS ESPECÍFICOS
                # ============================================
                st.header("🔍 Análise de Números Específicos")
                st.write("Digite os números que deseja analisar, separados por vírgula (ex: 5, 23, 42)")
                
                # O text_input agora funciona porque estamos dentro do bloco do session_state
                numeros_input = st.text_input(
                    "Números para análise:",
                    placeholder="Ex: 5, 23, 42, 7, 15"
                )
                
                if numeros_input:
                    try:
                        numeros_analisar = [int(n.strip()) for n in numeros_input.split(',') if n.strip()]
                        numeros_validos = [n for n in numeros_analisar if 1 <= n <= 60]
                        
                        if numeros_validos:
                            st.success(f"📌 Analisando os números: {', '.join(map(str, sorted(numeros_validos)))}")
                            
                            for numero in sorted(numeros_validos):
                                st.markdown(f"### 🎯 Número **{numero}**")
                                
                                total_aparicoes = (bolas_df == numero).sum().sum()
                                
                                ultima_aparicao_idx = None
                                for idx in range(len(df)-1, -1, -1):
                                    if numero in bolas_df.iloc[idx].values:
                                        ultima_aparicao_idx = idx
                                        break
                                
                                if ultima_aparicao_idx is not None:
                                    atraso = len(df) - 1 - ultima_aparicao_idx
                                    concurso_ultima = df.iloc[ultima_aparicao_idx, 0] if df.shape[1] > 0 else 'N/A'
                                else:
                                    atraso = len(df)
                                    concurso_ultima = "Nunca"
                                
                                ultimos_50 = bolas_df.tail(50)
                                aparicoes_50 = (ultimos_50 == numero).sum().sum()
                                ultimos_100 = bolas_df.tail(100)
                                aparicoes_100 = (ultimos_100 == numero).sum().sum()
                                ultimos_200 = bolas_df.tail(200)
                                aparicoes_200 = (ultimos_200 == numero).sum().sum()
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("✅ Total de Aparições", total_aparicoes)
                                with col2:
                                    st.metric("🕐 Última Aparição", f"Concurso {concurso_ultima}")
                                with col3:
                                    st.metric("⏳ Sorteios sem Aparecer", atraso)
                                with col4:
                                    freq_media = (total_aparicoes / len(df)) * 100
                                    st.metric("📊 Frequência", f"{freq_media:.2f}%")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("📅 Últimos 50 sorteios", f"{aparicoes_50}x")
                                with col2:
                                    st.metric("📅 Últimos 100 sorteios", f"{aparicoes_100}x")
                                with col3:
                                    st.metric("📅 Últimos 200 sorteios", f"{aparicoes_200}x")
                                
                                # Gráfico simples
                                aparicoes_tempo = []
                                janela = 50
                                for i in range(janela, len(df)+1):
                                    janela_df = bolas_df.iloc[i-janela:i]
                                    count = (janela_df == numero).sum().sum()
                                    aparicoes_tempo.append(count)
                                
                                if len(aparicoes_tempo) > 0:
                                    fig, ax = plt.subplots(figsize=(14, 4))
                                    ax.plot(range(janela, len(df)+1), aparicoes_tempo, color='steelblue', linewidth=2, alpha=0.7)
                                    ax.fill_between(range(janela, len(df)+1), aparicoes_tempo, alpha=0.3, color='steelblue')
                                    ax.set_title(f'Frequência do Número {numero} (Janela de 50 sorteios)')
                                    st.pyplot(fig)
                                
                                st.markdown("---")
                            
                            # Análise combinada
                            if len(numeros_validos) >= 2:
                                st.markdown("### 🔗 Análise Combinada")
                                st.write(f"Verificando quantas vezes os números apareceram **juntos**:")
                                
                                for i in range(2, min(len(numeros_validos) + 1, 7)):
                                    st.markdown(f"#### Combinações de {i} números:")
                                    for combo in combinations(sorted(numeros_validos), i):
                                        count = 0
                                        for _, row in bolas_df.iterrows():
                                            if all(num in row.values for num in combo):
                                                count += 1
                                        
                                        combo_str = '-'.join(map(str, combo))
                                        if count > 0:
                                            st.write(f"✅ **{combo_str}**: apareceram juntos **{count} vezes**")
                                        else:
                                            st.write(f"❌ **{combo_str}**: nunca apareceram juntos")
                        else:
                            st.warning("⚠️ Por favor, digite números válidos entre 1 e 60.")
                    except ValueError:
                        st.error("❌ Erro ao processar os números. Use o formato: 5, 23, 42")
                
                st.markdown("---")

                # ============================================
                # 1. FREQUÊNCIA GERAL (Restante do código...)
                # ============================================
                st.header("1️⃣ Frequência Geral dos Números")
                
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
                st.pyplot(fig)
                
                st.markdown("---")

                # ============================================
                # 2. PARES FREQUENTES
                # ============================================
                st.header("2️⃣ Pares que Mais Saíram Juntos")
                
                # (Mantive a lógica resumida para não estourar o limite, mas a estrutura está correta)
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
                ax.barh(pares_labels, pares_valores, color='teal', alpha=0.7)
                ax.invert_yaxis()
                st.pyplot(fig)

                st.markdown("---")
                
                # ============================================
                # AVISO FINAL
                # ============================================
                st.warning("⚠️ Padrões históricos NÃO aumentam as chances de prever resultados futuros.")

            except Exception as e:
                st.error(f"Erro: {e}")