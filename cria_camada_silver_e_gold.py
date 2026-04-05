import sys
import pandas as pd
import awswrangler as wr
import uuid

# =============================================================================
# CONFIGURAÇÕES DE AMBIENTE E CAMINHOS DO S3
# =============================================================================
S3_BRONZE_PATH = "s3://dressasys-sagemakers-fiap/analise_covid19/inteligencia_hospitalar_bronze/"
S3_SILVER_PATH = "s3://dressasys-sagemakers-fiap/analise_covid19/inteligencia_hospitalar_silver/"
S3_GOLD_PATH = "s3://dressasys-sagemakers-fiap/analise_covid19/inteligencia_hospitalar_gold/"

def processar_etl_pnad_covid():
    # =========================================================================
    # 1. SELEÇÃO DE VARIÁVEIS (Atualizada com Variáveis Temporais)
    # =========================================================================
    mapa_colunas = {
        # Metadados de Pesos e Geografia
        'UF': 'cod_uf', 'CAPITAL': 'cod_capital', 'RM_RIDE': 'cod_rm_ride',
        'V1031': 'peso_amostral_sem_calibracao', 'V1032': 'peso_amostral_com_calibracao',
        
        # Pilar Tempo
        'Ano': 'ano', 'V1013': 'mes_pesquisa',
        
        # Pilar 1: Demografia
        'A002': 'idade', 'A003': 'cod_sexo', 'A004': 'cod_cor_raca',
        
        # Pilar 2: Sintomas
        'B0011': 'teve_febre', 'B0012': 'teve_tosse', 
        'B0014': 'dificuldade_respirar', 'B00111': 'perda_cheiro_sabor',
        
        # Pilar 3: Comportamento
        'B002': 'buscou_atendimento', 'B011': 'medida_isolamento',
        
        # Pilar 4: Economia e Sociedade
        'C001': 'trabalhou_na_semana', 'C013': 'trabalho_remoto', 
        'C01012': 'rendimento_efetivo', 'D0051': 'auxilio_emergencial', 
        'F001': 'condicao_domicilio'
    }

    print("1. Iniciando leitura otimizada da Camada Bronze...")
    colunas_para_ler = list(mapa_colunas.keys())
    
    df_raw = wr.s3.read_csv(
        path=S3_BRONZE_PATH,
        dataset=True,
        usecols=colunas_para_ler
    )

    print("2. Renomeando colunas e gerando chaves primárias...")
    df = df_raw.rename(columns=mapa_colunas).copy()
    
    # Gerando Chave Única (Surrogate Key) para conectar a Fato com as Dimensões
    df['id_respondente'] = [str(uuid.uuid4()) for _ in range(len(df))]

    print("3. Construindo Tabelas Dimensionais (Camada Silver)...")

    # =========================================================================
    # NOVA DIMENSÃO 0: TEMPO
    # =========================================================================
    df['id_tempo'] = df['ano'].astype(str) + "-" + df['mes_pesquisa'].astype(str).str.zfill(2)
    dim_tempo = df[['id_tempo', 'ano', 'mes_pesquisa']].drop_duplicates().copy()
    
    mapa_meses = {9: 'Setembro', 10: 'Outubro', 11: 'Novembro'}
    dim_tempo['nome_mes'] = dim_tempo['mes_pesquisa'].map(mapa_meses).fillna('Desconhecido')

    # =========================================================================
    # DIMENSÃO 1: LOCALIDADE
    # =========================================================================
    mapa_uf = {
        11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
        21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA',
        31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC', 43: 'RS',
        50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
    }
    dim_localidade = df[['id_respondente', 'cod_uf', 'cod_capital', 'cod_rm_ride']].copy()
    dim_localidade['nome_uf'] = dim_localidade['cod_uf'].map(mapa_uf)
    
    def classificar_polo(row):
        if pd.notna(row['cod_capital']): return f"Capital ({row['nome_uf']})"
        elif pd.notna(row['cod_rm_ride']): return f"Região Metropolitana ({row['nome_uf']})"
        else: return f"Interior ({row['nome_uf']})"
        
    dim_localidade['polo_epidemiologico'] = dim_localidade.apply(classificar_polo, axis=1)
    dim_localidade = dim_localidade.drop(columns=['cod_capital', 'cod_rm_ride'])

    # =========================================================================
    # DIMENSÃO 2: DEMOGRAFIA
    # =========================================================================
    dim_demo = df[['id_respondente', 'idade', 'cod_sexo', 'cod_cor_raca']].copy()
    dim_demo['desc_sexo'] = dim_demo['cod_sexo'].map({1: 'Homem', 2: 'Mulher'}).fillna('Ignorado')
    dim_demo['desc_cor_raca'] = dim_demo['cod_cor_raca'].map({1: 'Branca', 2: 'Preta', 3: 'Amarela', 4: 'Parda', 5: 'Indígena', 9: 'Ignorado'}).fillna('Ignorado')
    dim_demo['faixa_etaria'] = pd.cut(dim_demo['idade'], bins=[0, 18, 30, 50, 65, 120], labels=['0-18', '19-30', '31-50', '51-65', '65+'], right=False).astype(str)

    # =========================================================================
    # DIMENSÃO 3: SINTOMAS CLÍNICOS
    # =========================================================================
    dim_sint = df[['id_respondente', 'teve_febre', 'teve_tosse', 'dificuldade_respirar', 'perda_cheiro_sabor']].copy()
    mapa_sim_nao = {1.0: 'Sim', 2.0: 'Não', 3.0: 'Não Sabe', 9.0: 'Não Aplicável'}
    for col in ['teve_febre', 'teve_tosse', 'dificuldade_respirar', 'perda_cheiro_sabor']:
        dim_sint[col] = dim_sint[col].fillna(9.0)
        dim_sint[f'desc_{col}'] = dim_sint[col].map(mapa_sim_nao)

    # =========================================================================
    # DIMENSÃO 4: COMPORTAMENTO
    # =========================================================================
    dim_comp = df[['id_respondente', 'buscou_atendimento', 'medida_isolamento']].copy()
    dim_comp['buscou_atendimento'] = dim_comp['buscou_atendimento'].fillna(9.0)
    dim_comp['desc_buscou_atendimento'] = dim_comp['buscou_atendimento'].map(mapa_sim_nao)
    
    mapa_isolamento = {
        1.0: 'Não fez restrição', 2.0: 'Reduziu contato', 
        3.0: 'Ficou em casa (saiu p/ o básico)', 4.0: 'Rigorosamente isolado em casa', 9.0: 'Não Aplicável'
    }
    dim_comp['medida_isolamento'] = dim_comp['medida_isolamento'].fillna(9.0)
    dim_comp['desc_medida_isolamento'] = dim_comp['medida_isolamento'].map(mapa_isolamento).fillna('Não Aplicável')

    # =========================================================================
    # DIMENSÃO 5: ECONOMIA E SOCIEDADE
    # =========================================================================
    dim_econ = df[['id_respondente', 'trabalhou_na_semana', 'trabalho_remoto', 'rendimento_efetivo', 'auxilio_emergencial', 'condicao_domicilio']].copy()
    dim_econ['desc_trabalhou_semana'] = dim_econ['trabalhou_na_semana'].map({1.0: 'Sim', 2.0: 'Não'}).fillna('Não Aplicável')
    dim_econ['desc_trabalho_remoto'] = dim_econ['trabalho_remoto'].map({1.0: 'Sim', 2.0: 'Não'}).fillna('Não Aplicável')
    dim_econ['desc_auxilio_emergencial'] = dim_econ['auxilio_emergencial'].map({1.0: 'Sim', 2.0: 'Não'}).fillna('Não Aplicável')
    
    mapa_domicilio = {1.0: 'Próprio - já pago', 2.0: 'Próprio - pagando', 3.0: 'Alugado', 4.0: 'Cedido por empregador', 5.0: 'Cedido por familiar', 6.0: 'Cedido de outra forma', 7.0: 'Outra condição'}
    dim_econ['desc_condicao_domicilio'] = dim_econ['condicao_domicilio'].map(mapa_domicilio).fillna('Não Aplicável')
    dim_econ['rendimento_efetivo'] = dim_econ['rendimento_efetivo'].fillna(0)

    # =========================================================================
    # TABELA FATO (Métricas e Chaves atualizadas com a nova dimensão)
    # =========================================================================
    print("4. Construindo Tabela Fato...")
    fato_covid = df[['id_respondente', 'id_tempo', 'peso_amostral_com_calibracao']].copy()

    # =========================================================================
    # EXPORTAÇÃO PARA CAMADA SILVER NO S3 (FORMATO PARQUET)
    # =========================================================================
    print("5. Salvando arquivos da Camada Silver (Parquet)...")
    
    tabelas_silver = {
        "Fato_Respostas_Covid": fato_covid,
        "Dim_Tempo": dim_tempo,
        "Dim_Localidade": dim_localidade,
        "Dim_Demografia": dim_demo,
        "Dim_Sintomas": dim_sint,
        "Dim_Comportamento": dim_comp,
        "Dim_Economia": dim_econ
    }

    for nome_tabela, dataframe in tabelas_silver.items():
        caminho_tabela = f"{S3_SILVER_PATH}{nome_tabela}/"
        wr.s3.to_parquet(
            df=dataframe,
            path=caminho_tabela,
            dataset=True,
            mode="overwrite"
        )

    # =========================================================================
    # 6. CRIAÇÃO E EXPORTAÇÃO DA CAMADA GOLD (Agregações de Negócio)
    # =========================================================================
    print("6. Construindo a Camada Gold (Visões Agregadas para o Painel)...")

    # Mesclando as tabelas da camada Silver (agora incluindo o Tempo)
    df_gold_base = fato_covid.merge(dim_tempo, on='id_tempo', how='inner') \
                             .merge(dim_localidade, on='id_respondente', how='inner') \
                             .merge(dim_demo, on='id_respondente', how='inner') \
                             .merge(dim_sint, on='id_respondente', how='inner') \
                             .merge(dim_comp, on='id_respondente', how='inner') \
                             .merge(dim_econ, on='id_respondente', how='inner')

    # Produto Gold 1: Alerta de Risco Hospitalar
    gold_risco_hospitalar = df_gold_base[
        (df_gold_base['desc_dificuldade_respirar'] == 'Sim') | 
        (df_gold_base['desc_teve_febre'] == 'Sim')
    ].groupby(['id_tempo', 'nome_mes', 'polo_epidemiologico', 'faixa_etaria', 'desc_dificuldade_respirar', 'desc_teve_febre']) \
     .agg(total_pacientes_estimados=('peso_amostral_com_calibracao', 'sum')) \
     .reset_index()
    
    gold_risco_hospitalar['total_pacientes_estimados'] = gold_risco_hospitalar['total_pacientes_estimados'].round(0)

    # Produto Gold 2: Vulnerabilidade e Quebra de Isolamento
    gold_vulnerabilidade = df_gold_base.groupby([
        'id_tempo', 'nome_mes',
        'polo_epidemiologico', 
        'desc_medida_isolamento', 
        'desc_auxilio_emergencial', 
        'desc_trabalho_remoto'
    ]).agg(
        total_populacao=('peso_amostral_com_calibracao', 'sum'),
        renda_media_efetiva=('rendimento_efetivo', 'mean')
    ).reset_index()

    gold_vulnerabilidade['total_populacao'] = gold_vulnerabilidade['total_populacao'].round(0)
    gold_vulnerabilidade['renda_media_efetiva'] = gold_vulnerabilidade['renda_media_efetiva'].round(2)

    # -------------------------------------------------------------------------
    # NOVO PRODUTO GOLD 3: Perfil Sintomático (Melt)
    # -------------------------------------------------------------------------
    colunas_id = ['id_tempo', 'nome_mes', 'polo_epidemiologico', 'faixa_etaria', 'peso_amostral_com_calibracao']
    colunas_sintomas = ['desc_teve_febre', 'desc_teve_tosse', 'desc_dificuldade_respirar', 'desc_perda_cheiro_sabor']

    df_sintomas_melt = df_gold_base[colunas_id + colunas_sintomas].melt(
        id_vars=colunas_id,
        value_vars=colunas_sintomas,
        var_name='nome_sintoma',
        value_name='resposta_paciente'
    )

    df_sintomas_sim = df_sintomas_melt[df_sintomas_melt['resposta_paciente'] == 'Sim'].copy()

    mapa_nomes_sintomas = {
        'desc_teve_febre': 'Febre',
        'desc_teve_tosse': 'Tosse',
        'desc_dificuldade_respirar': 'Falta de Ar',
        'desc_perda_cheiro_sabor': 'Perda de Olfato/Paladar'
    }
    df_sintomas_sim['nome_sintoma'] = df_sintomas_sim['nome_sintoma'].map(mapa_nomes_sintomas)

    gold_perfil_sintomas = df_sintomas_sim.groupby(
        ['id_tempo', 'nome_mes', 'polo_epidemiologico', 'faixa_etaria', 'nome_sintoma']
    ).agg(
        total_pacientes_estimados=('peso_amostral_com_calibracao', 'sum')
    ).reset_index()

    gold_perfil_sintomas['total_pacientes_estimados'] = gold_perfil_sintomas['total_pacientes_estimados'].round(0)

    # =========================================================================
    # EXPORTAÇÃO DA CAMADA GOLD
    # =========================================================================
    print("7. Salvando arquivos da Camada Gold (Parquet)...")
    
    tabelas_gold = {
        "Gold_Alerta_Risco_Hospitalar": gold_risco_hospitalar,
        "Gold_Vulnerabilidade_Isolamento": gold_vulnerabilidade,
        "Gold_Perfil_Sintomas": gold_perfil_sintomas
    }

    for nome_tabela, dataframe in tabelas_gold.items():
        if 'total_pacientes_estimados' in dataframe.columns:
            dataframe = dataframe[dataframe['total_pacientes_estimados'] > 0]
        else:
            dataframe = dataframe[dataframe['total_populacao'] > 0]
            
        caminho_tabela = f"{S3_GOLD_PATH}{nome_tabela}/"
        wr.s3.to_parquet(
            df=dataframe,
            path=caminho_tabela,
            dataset=True,
            mode="overwrite"
        )

    print("=== Pipeline executado com SUCESSO! Dados modelados e armazenados. ===")

if __name__ == "__main__":
    processar_etl_pnad_covid()