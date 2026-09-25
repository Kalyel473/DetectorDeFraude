# FraudShield BR

**Detector de fraude com Machine Learning em tres dominios: transacoes financeiras, phishing em e-mail e perfis/posts fraudulentos em redes sociais.**

Ferramenta **defensiva**, feita para times de seguranca e antifraude e para o publico do canal *Cybersegurança na Prática*. Todo o pipeline e visivel: engenharia de features, tratamento de desbalanceamento, metricas honestas e principalmente  **a explicacao de por que o modelo classificou algo como fraude**.

```
[FRAUDSHIELD BR] Analisando transacao #6060...
→ Score de fraude: 1.00 (ALTO RISCO)
→ Principais fatores:
    • horario incomum (02:39)
    • transacao na madrugada (00h-06h)
    • componente PCA V14 = -6.07 (padrao anonimizado associado a fraude)
    • velocidade de transacao (3 transacoes na ultima hora)
→ Recomendacao: bloquear + verificacao manual
```

---

## Indice

- [Principio de design](#principio-de-design)
- [Instalacao](#instalacao)
- [Comecando em 30 segundos](#comecando-em-30-segundos)
- [Arquitetura](#arquitetura)
- [Fluxo do pipeline](#fluxo-do-pipeline)
- [Modulo 1 - transacoes](#modulo-1---deteccao-de-fraude-em-transacoes)
- [Modulo 2 - phishing em e-mail](#modulo-2---deteccao-de-phishing-em-e-mail)
- [Modulo 3 - redes sociais](#modulo-3---deteccao-de-fraude-em-redes-sociais)
- [Metricas: por que accuracy engana](#metricas-por-que-accuracy-engana)
- [Explicabilidade (SHAP)](#explicabilidade-shap)
- [Camada BERT opcional](#camada-bert-opcional)
- [Referencia da CLI](#referencia-da-cli)
- [Usando os datasets reais](#usando-os-datasets-reais)
- [Sobre as amostras do repositorio](#sobre-as-amostras-do-repositorio)
- [Limitacoes conhecidas](#limitacoes-conhecidas)
- [Etica e aviso legal](#etica-e-aviso-legal)

---

## Principio de design

Cada modulo tem um modo `--demo` que roda o pipeline inteiro com dados de exemplo **ja incluidos no repositorio**: sem download do Kaggle, sem API paga, sem GPU obrigatoria. O fine-tuning de BERT e **plugavel e opcional** (`--model bert`), nunca requisito.

Tres regras que a ferramenta nunca quebra:

1. **SMOTE so no treino.** Sintetizar minoria antes do split vaza informacao e infla a metrica.
2. **Nenhuma decisao sem explicacao.** Score sozinho nao vira acao; o motivo vem junto.
3. **Degradacao controlada.** Falta `shap`? Explica por oclusao. Falta `imbalanced-learn`? Usa SMOTE interno. Falta `xgboost`? Cai pro GradientBoosting, avisando.

---

## Instalacao

Python 3.11 ou superior (testado em 3.14).

```bash
git clone <seu-repo> fraudshield-br
cd fraudshield-br
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

O nucleo obrigatorio e pequeno: `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib`, `matplotlib`, `rich`. Os opcionais (`transformers`, `torch`, `xgboost`, `python-whois`) estao comentados no `requirements.txt`.

As amostras do modo demo ja vem versionadas em `data/samples/`. Para regerar:

```bash
python scripts/generate_samples.py
```

---

## Comecando em 30 segundos

```bash
# pipeline completo nos tres dominios (treino + metricas + SHAP + predicao explicada)
python main.py --module transaction --demo
python main.py --module email --demo
python main.py --module social --demo

# classificar um e-mail real (.eml salvo do seu cliente de e-mail)
python main.py --module email --predict data/samples/exemplo_phishing.eml
python main.py --module email --predict data/samples/exemplo_legitimo.eml

# classificar transacoes / perfis a partir de CSV ou JSON
python main.py --module transaction --predict data/samples/transacoes_suspeitas.json
python main.py --module social --predict data/samples/perfis_suspeitos.json

# relatorio de metricas do modelo salvo
python main.py --module transaction --evaluate

# tabela comparativa entre modelos
python main.py --module email --compare rf,gb,logreg
```

Cada execucao grava em `reports/<modulo>/`: `roc.png`, `precision_recall.png`, `matriz_confusao.png`, `importancia_global.png`, `metricas.json` e `relatorio.md`. Os modelos treinados vao para `models/<modulo>_<modelo>.joblib`.

---

## Arquitetura

```
fraudshield-br/
├── data/
│   ├── raw/                        # datasets brutos (nao versionados, .gitignore)
│   ├── processed/                  # dados intermediarios
│   └── samples/                    # amostras versionadas do modo --demo
├── src/
│   ├── modules/
│   │   ├── base.py                 # contrato comum dos modulos
│   │   ├── transaction_fraud.py    # modulo 1: transacoes
│   │   ├── email_phishing.py       # modulo 2: e-mail (+ parser .eml)
│   │   └── social_fraud.py         # modulo 3: perfis/posts
│   ├── features/
│   │   ├── transaction_features.py # valor, velocidade, horario, PCA V1-V28
│   │   ├── text_features.py        # TF-IDF, gatilhos, URL, cabecalho, WHOIS
│   │   └── social_features.py      # conta, texto, rede (DBSCAN de criacao)
│   ├── models/
│   │   ├── baseline_sklearn.py     # RF / GradientBoosting / XGBoost / LogReg + SMOTE
│   │   ├── bert_classifier.py      # fine-tuning opcional (BERTimbau/DistilBERT)
│   │   └── explainability.py       # SHAP + fallback por oclusao
│   ├── pipeline/
│   │   ├── train.py                # dados -> features -> split -> SMOTE -> fit
│   │   ├── evaluate.py             # metricas, curvas, relatorios
│   │   └── predict.py              # score + fatores + recomendacao
│   ├── cli.py                      # argparse, acoes e subcomandos
│   ├── config.py                   # caminhos, seeds, faixas de risco
│   └── ui.py                       # saida no terminal (estilo Terminal Verde)
├── models/                         # modelos serializados (.joblib / dir do BERT)
├── reports/                        # metricas, curvas e graficos gerados
├── notebooks/                      # EDA exploratoria (opcional)
├── scripts/generate_samples.py     # gerador das amostras do --demo
├── main.py
├── requirements.txt
└── README.md
```

Os tres modulos implementam o mesmo contrato (`src/modules/base.py`): carregar dados, construir features, identificar um item e **traduzir o nome tecnico de uma feature para portugues de gente**. Por isso o pipeline (`train`/`evaluate`/`predict`) e totalmente generico — adicionar um quarto dominio e escrever uma classe nova, nao um pipeline novo.

---

## Fluxo do pipeline

```
dados brutos
   ↓ engenharia de features (por dominio)
   ↓ split treino/teste ESTRATIFICADO  ← antes do balanceamento, sempre
   ↓ SMOTE (so no treino)
   ↓ treinamento (RF / GB / XGB / LogReg / BERT)
   ↓ avaliacao (Precision, Recall, F1, ROC-AUC, PR-AUC, matriz de confusao)
   ↓ explicabilidade (SHAP global + individual)
   → predicao final: score + motivo + recomendacao
```

---

## Modulo 1 - Deteccao de fraude em transacoes

**Dataset de referencia:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group / Kaggle) — 284.807 transacoes, 492 fraudes (**0,172%**). Alternativa com mais features categoricas: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection).

**Features:**

| feature | o que captura |
| --- | --- |
| `amount`, `amount_log` | valor da transacao (log domina a cauda longa) |
| `hour_of_day`, `is_night` | hora derivada de `Time`; madrugada e janela classica de teste de cartao |
| `secs_since_last_tx` | tempo desde a transacao anterior **do mesmo cartao** |
| `tx_count_last_hour` | frequencia na ultima hora (velocidade) |
| `amount_ratio_card_mean` | desvio do valor medio historico do cartao |
| `amount_zscore_global` | desvio em relacao a carteira toda (media/desvio fixados no treino) |
| `velocity_flag` | 3+ transacoes em 2 minutos |
| `V1`-`V28` | componentes PCA ja anonimizados no dataset |

> **Sobre V1-V28:** sao componentes PCA, ou seja, **dados sensiveis mascarados**. Isso nao e limitacao do dataset, e a pratica real de compliance (PCI-DSS, LGPD): o modelo aprende o padrao sem que ninguem veja o dado cru do titular. O custo e a explicabilidade — "V14 muito negativo" nao vira frase de negocio sozinho, por isso as features derivadas (valor, horario, velocidade) carregam o peso da explicacao.

As features de velocidade usam apenas o **passado** de cada cartao (nada de olhar o futuro, que seria leakage temporal). Sem coluna de cartao, a serie e tratada como fluxo unico, com aviso.

**Modelos:** `RandomForestClassifier(n_estimators=200, class_weight="balanced")` como baseline, comparavel com `GradientBoostingClassifier`, `XGBClassifier` e `LogisticRegression`. SMOTE aplicado **somente no treino**.

---

## Modulo 2 - Deteccao de phishing em e-mail

**Datasets de referencia:** [Enron Email Dataset](https://www.cs.cmu.edu/~enron/) (legitimos) combinado com o [Nazario Phishing Corpus](https://monkey.org/~jose/phishing/) ou feeds do [PhishTank](https://phishtank.org/) (fraudulentos) — todos publicos e amplamente usados em pesquisa academica.

**Tres blocos de sinal, propositalmente separados para ficarem explicaveis:**

1. **Texto** — TF-IDF (1-2 gramas, acentos normalizados, stopwords PT+EN) + contagem de palavras-gatilho por categoria: `urgencia` ("urgente", "ultimo aviso"), `credencial` ("clique aqui", "verifique agora", "confirme seus dados"), `ameaca` ("sua conta sera bloqueada"), `dinheiro` ("pix", "restituicao", "premio"), `anexo`.
2. **URL** — IP puro no lugar do dominio, encurtador, TLD de alto abuso (`.top`, `.zip`, `.xyz`...), punycode/homoglifos, `@` antes do host, **typosquatting por distancia de Levenshtein** contra 29 marcas brasileiras (`nubarnk.com` → distancia 1 de `nubank`) e **marca legitima em dominio de terceiro** (`itau.verifica-cliente.top`).
3. **Cabecalho** — divergencia `From` x `Reply-To`, SPF/DKIM invalido ou ausente, marca no nome exibido com dominio real diferente, remetente freemail se passando por marca, **idade do dominio** (coluna `domain_age_days` ou consulta WHOIS ao vivo com `--whois`).

**Entrada:** CSV/JSON com as colunas `from_name, from_addr, reply_to, subject, body, spf_result, dkim_result, attachments, domain_age_days, label` — ou um **`.eml` real**, que o parser abre e converte (cabecalhos, corpo texto/HTML, anexos, `Authentication-Results`).

**Modelos:** `TfidfVectorizer` + `RandomForest` (padrao) ou `LogisticRegression`. Camada opcional: BERTimbau fine-tunado (`--model bert`).

---

## Modulo 3 - Deteccao de fraude em redes sociais

**Dataset de referencia:** [Cresci et al.](https://botometer.osome.iu.edu/bot-repository/datasets.html) (fake followers / social spambots), referencia academica classica da area.

**Features:**

- **Comportamentais da conta:** idade da conta, razao seguidores/seguindo, posts por dia, seguidores por dia, foto de perfil padrao, biografia vazia, proporcao de digitos no `@` (padrao de conta gerada em massa), digitos no nome de exibicao.
- **Texto do post:** TF-IDF, termos de golpe ("renda extra", "link na bio", "lucro garantido", "airdrop"), links encurtados, TLD suspeito, CAPS, exclamacoes, emojis, hashtags e mencoes em excesso.
- **Rede:** `duplicate_text_count` (mesma mensagem publicada por N contas = spam coordenado) e `creation_burst_size` — **clustering temporal com DBSCAN** sobre o timestamp de criacao, que expoe fazendas de bots criadas na mesma janela de 6 horas.

Os dois sinais de rede sao **de lote**. Quando a predicao recebe uma conta isolada, o FraudShield recorre a memoria construida no treino (avisando na saida) — a alternativa honesta a inventar contexto que nao existe.

---

## Metricas: por que accuracy engana

No dataset da ULB, 0,172% das transacoes sao fraude. O modelo preguicoso que responde "legitimo" para **tudo**:

```
accuracy = 284.315 / 284.807 = 99,83%
recall   = 0 / 492         = 0%      ← nao detectou uma unica fraude
```

99,83% de accuracy e **zero** utilidade. Por isso o FraudShield sempre imprime, junto:

| metrica | leitura |
| --- | --- |
| **Precision** | dos alarmes disparados, quantos eram fraude (custo da fila de revisao) |
| **Recall** | das fraudes existentes, quantas foram pegas (prejuizo evitado) |
| **F1** | equilibrio entre os dois |
| **ROC-AUC** | separacao geral entre as classes |
| **PR-AUC** | **a metrica principal em classe rara** — nao se deixa enganar pela massa de negativos |
| **Matriz de confusao** | TN / FP / FN / TP em numeros absolutos |
| **Melhor limiar** | o corte que maximiza F1, em vez de aceitar 0.5 por inercia |

A saida traz tambem a `accuracy` do chute majoritario ao lado da accuracy do modelo — a comparacao que desmonta o numero bonito.

**Resultados nas amostras do repositorio** (`--demo`, seed 42, split 75/25):

| modulo | modelo | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| transaction | RandomForest | 0,9167 | 0,6471 | 0,7586 | 0,8889 | 0,9969 |
| transaction | GradientBoosting | 0,8333 | 0,8824 | 0,8571 | 0,8993 | 0,9935 |
| transaction | LogisticRegression | 0,8824 | 0,8824 | 0,8824 | 0,9182 | 0,9975 |
| email | RandomForest | 1,0000 | 0,9057 | 0,9505 | 0,9437 | 0,9401 |
| email | LogisticRegression | 1,0000 | 0,8679 | 0,9293 | 0,9571 | 0,9648 |
| social | RandomForest | 0,9836 | 0,8000 | 0,8824 | 0,9088 | 0,9514 |

Repare no caso `transaction`/RandomForest: PR-AUC 0,89 com recall 0,65 no limiar 0,5. O modelo **ordena** bem as fraudes, mas o corte padrao esta conservador — exatamente a conversa de limiar que aparece em `--evaluate` (o melhor F1 vem em ~0,40, onde o F1 sobe para 0,88). Numeros de amostra sintetica; veja a secao seguinte.

---

## Explicabilidade (SHAP)

Duas camadas:

**Global** — `shap.TreeExplainer` para floresta/boosting (media do `|valor SHAP|` por feature), com grafico em `reports/<modulo>/<modelo>_importancia_global.png`. A tabela no terminal mostra o **valor tipico da feature entre as fraudes do treino**, ja traduzido:

```
┌────┬──────────────────────┬────────────┬────────────────────────────────────┐
│ #  │ feature              │ influencia │ leitura (valor tipico nas fraudes) │
├────┼──────────────────────┼────────────┼────────────────────────────────────┤
│ 1  │ username_digit_ratio │ 0.05460    │ 58% do @ sao digitos (padrao de    │
│    │                      │            │ conta gerada em massa)             │
│ 2  │ scam_triggers        │ 0.03583    │ 3 termo(s) de golpe no post        │
│ 4  │ creation_burst_size  │ 0.02618    │ 28 contas criadas na mesma janela  │
│    │                      │            │ de 6h (fazenda de bots)            │
└────┴──────────────────────┴────────────┴────────────────────────────────────┘
```

**Individual** — valores SHAP da linha, ordenados por contribuicao, traduzidos pelo modulo e exibidos no painel de score. Quando o veredito e negativo, o titulo muda para *"Sinais observados (nenhum decisivo)"* — o painel nunca sugere que sinais fracos condenaram alguem.

**Sem `shap` instalado**, a explicacao individual usa **oclusao**: cada feature candidata e substituida pelo valor tipico do treino e mede-se a queda no score. Menos rigoroso que Shapley, rapido e sempre disponivel — e a saida informa qual metodo foi usado (`explicacao via: SHAP` / `oclusao`). Para o BERT, a oclusao e por palavra.

Mesmo padrao de transparencia do ImunoShield (formula de risco visivel) e do EscalaMind (regras explicaveis).

---

## Camada BERT opcional

```bash
pip install transformers torch
python main.py --module email --train --model bert --bert-epochs 2 --bert-batch 16
python main.py --module email --predict data/samples/exemplo_phishing.eml --model bert
```

- Modelo padrao: **BERTimbau** (`neuralmind/bert-base-portuguese-cased`). Alternativa leve: `--bert-model distilbert-base-multilingual-cased`.
- Roda na RTX 4060 com `batch_size=16` e **fp16 automatico** quando ha CUDA (`torch.amp`), caindo para fp32 na CPU.
- Desbalanceamento em texto **nao usa SMOTE** (interpolar embeddings gera frase que nao existe): usa peso de classe na funcao de perda.
- `BertTextClassifier.embed()` expoe o vetor `[CLS]`, permitindo reaproveitar o modelo do modulo de e-mail como fonte de features no modulo social.
- Explicacao por oclusao de palavras: remove um termo por vez e mede a queda no score.

O `--compare` imprime baseline x BERT com tempo de treino e custo computacional lado a lado — o argumento honesto e que o baseline resolve em segundos na CPU, e o transformer cobra GPU e minutos para ganhar alguns pontos de recall no texto.

---

## Referencia da CLI

```
python main.py --module {transaction|email|social} [acao] [opcoes]
```

**Acoes**

| flag | efeito |
| --- | --- |
| `--demo` | pipeline completo com a amostra do repositorio + predicoes explicadas |
| `--train` | treina e salva o modelo em `models/` |
| `--evaluate` | recalcula metricas do modelo salvo e gera relatorio |
| `--predict ARQUIVO` | classifica `.eml`, `.csv`, `.json`, `.jsonl` ou `.parquet` |
| `--compare [rf,gb,logreg,xgb,bert]` | treina varios e imprime a tabela comparativa |
| `--list-modules` | lista os dominios disponiveis |

**Dados e modelo**

| flag | padrao | efeito |
| --- | --- | --- |
| `--data ARQUIVO` | — | dataset de treino (senao usa `data/raw/` ou a amostra) |
| `--model` | `rf` | `rf`, `gb`, `xgb`, `logreg`, `bert` |
| `--no-smote` | off | desliga o SMOTE (util para mostrar o efeito dele) |
| `--test-size` | `0.25` | proporcao do teste |
| `--seed` | `42` | semente (split e modelo reproduzivel) |
| `--whois` | off | modulo email: consulta WHOIS real da idade do dominio |

**Saida**

| flag | padrao | efeito | 
| --- | --- | --- |
| `--factors N` | `5` | fatores por predicao |
| `--top-k N` | `15` | features na importancia global |
| `--limit N` | `5` | itens explicados (`0` = todos) |
| `--threshold F` | `0.5` | limiar de decisao |
| `--only-flagged` | off | mostra apenas itens acima do limiar |
| `--json ARQUIVO` | — | salva o resultado da predicao em JSON |
| `--no-plots` / `--no-shap` / `--no-banner` | off | saida mais rapida/enxuta |

**BERT:** `--bert-epochs`, `--bert-batch`, `--bert-model`.

---

## Usando os datasets reais

```bash
# 1. transacoes: baixe creditcard.csv do Kaggle
mv ~/Downloads/creditcard.csv data/raw/
python main.py --module transaction --train            # detecta data/raw/creditcard.csv
python main.py --module transaction --train --data data/raw/creditcard.csv   # explicito

# 2. e-mail: monte um CSV com Enron (label 0) + Nazario/PhishTank (label 1)
python main.py --module email --train --data data/raw/emails.csv

# 3. social: dataset Cresci et al.
python main.py --module social --train --data data/raw/social.csv
```

Nomes procurados automaticamente em `data/raw/`: `creditcard.csv`, `transacoes.csv`, `train_transaction.csv`; `emails.csv`, `phishing.csv`, `enron_nazario.csv`; `social.csv`, `twitter_accounts.csv`, `cresci.csv`.

Colunas de rotulo aceitas: `Class`, `label`, `is_fraud`, `isFraud`, `fraude`, `is_phishing`, `is_fake`, `bot` (1 = fraude). Colunas ausentes degradam com aviso em vez de quebrar.

`data/raw/` e `data/processed/` estao no `.gitignore`: dado real nao entra no repositorio.

---

## Sobre as amostras do repositorio

`data/samples/` contem dados **sinteticos** gerados por `scripts/generate_samples.py`:

| arquivo | conteudo |
| --- | --- |
| `transacoes_amostra.csv` | 3.968 transacoes, 68 fraudes (1,71%), formato da ULB + `card_id` |
| `emails_amostra.csv` | 970 e-mails, 210 phishing (21,6%) |
| `social_amostra.csv` | 2.600 contas, 300 fraudulentas (11,5%) |
| `exemplo_phishing.eml` / `exemplo_legitimo.eml` | e-mails avulsos para `--predict` |
| `transacoes_suspeitas.json` / `perfis_suspeitos.json` | entradas avulsas para `--predict` |

As distribuicoes foram **deliberadamente sobrepostas**: parte das fraudes se disfarca (BEC sem link e com SPF valido, bot maduro com bio e foto, fraude em horario comercial) e parte dos registros legitimos parece suspeita (compra de madrugada, SPF mal configurado, conta nova sem bio, pequeno negocio divulgando promocao). Sem essa sobreposicao o modelo acerta 100% e a demonstracao fica irreal — **falso positivo e falso negativo precisam aparecer na tela**.

Ainda assim: **as metricas obtidas sobre as amostras sao ilustrativas**, servem para mostrar o pipeline funcionando. Numero para publicar em relatorio sai dos datasets reais.

---

## Limitacoes conhecidas

- **Velocidade precisa de historico.** Ao classificar 1-3 transacoes isoladas, `secs_since_last_tx`, `tx_count_last_hour` e `amount_ratio_card_mean` saem neutras e o score fica conservador. A ferramenta avisa. Em producao, alimente a janela de historico do cartao junto com a transacao.
- **Sinais de rede sao de lote.** `creation_burst_size` e `duplicate_text_count` dependem do conjunto; na predicao unitaria o FraudShield usa a memoria do treino.
- **WHOIS depende de rede e rate limit.** Por isso `--whois` e opt-in; o caminho recomendado e enriquecer a coluna `domain_age_days` no seu pipeline de ingestao e deixar o modelo consumir o valor em cache.
- **Limiar 0.5 raramente e o ideal.** Use o campo "Melhor limiar" e `--threshold` para calibrar conforme a capacidade da fila de revisao.
- **Drift.** Fraude muda de padrao; retreine periodicamente e acompanhe a PR-AUC ao longo do tempo.
- **TF-IDF nao entende semantica nova.** Golpe com texto inedito passa mais facil — e o argumento para a camada BERT.

---

## Etica e aviso legal

Esta e uma ferramenta de **uso exclusivamente defensivo**, destinada a times de seguranca, antifraude e integridade, e a fins educacionais.

- Os datasets usados na demonstracao sao **sinteticos**; os datasets de referencia recomendados sao **publicos e anonimizados** (componentes PCA, no caso da ULB).
- Em cenario real de producao, o tratamento de dados pessoais esta sujeito a **LGPD (Lei 13.709/2018)**: base legal definida, minimizacao, retencao limitada, registro das operacoes e direito de revisao de decisoes automatizadas (art. 20) — na pratica, todo alerta deve chegar ao analista com o motivo, e nunca bloquear um cliente sem caminho de contestacao humana.
- Acesso nao autorizado a dispositivo ou dado de terceiros e crime no Brasil: **Lei 12.737/2012** (Lei Carolina Dieckmann), que alterou o Codigo Penal. Detectar fraude nao autoriza investigar pessoas; use apenas dados que voce tem direito legitimo de processar.
- Nao use este projeto para perfilamento discriminatorio, vigilancia de individuos ou qualquer finalidade ofensiva.



