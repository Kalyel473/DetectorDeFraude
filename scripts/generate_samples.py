"""Gera as amostras versionadas de data/samples/ usadas pelo modo --demo.

Sao dados SINTETICOS, criados so para a demonstracao rodar sem depender de
download do Kaggle, de API paga ou de GPU. As metricas obtidas sobre eles sao
ilustrativas: para numeros de verdade use os datasets publicos reais
(creditcard.csv da ULB, Enron + Nazario, Cresci et al.) com --data.

As distribuicoes foram propositalmente SOBREPOSTAS: parte das fraudes se
disfarca de legitima (BEC sem link, bot bem feito, fraude em horario comercial)
e parte dos registros legitimos parece suspeita (compra de madrugada, SPF mal
configurado, conta nova sem bio). Sem essa sobreposicao o modelo acerta 100% e
a demonstracao fica irreal - falso positivo e falso negativo tem que aparecer.

Uso:  python scripts/generate_samples.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "samples"
SEED = 7
TODAY = datetime(2026, 9, 20, 12, 0, 0)


# --------------------------------------------------------------------------- #
# 1. Transacoes (formato compativel com o dataset da ULB: Time, V1..V28, Amount)
# --------------------------------------------------------------------------- #
def gen_transactions(n_legit: int = 3900, n_fraud: int = 68) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n_cards = 70
    rows = []

    for i in range(n_legit):
        card = int(rng.integers(0, n_cards))
        day = int(rng.integers(0, 2))
        if rng.random() < 0.06:                      # compra legitima de madrugada
            hour = float(rng.uniform(0.1, 5.8))
        else:
            hour = float(np.clip(rng.normal(14.5, 3.6), 6.2, 23.7))
        time = day * 86400 + hour * 3600 + rng.uniform(0, 120)
        if rng.random() < 0.03:                      # compra grande legitima (viagem, eletro)
            amount = float(np.clip(rng.lognormal(6.0, 0.6), 300, 9000))
        else:
            amount = float(np.clip(rng.lognormal(3.3, 0.85), 1.0, 2500))
        v = rng.normal(0, 1, 28)
        if rng.random() < 0.015:                     # cauda: legitima com cara de anomalia
            v *= rng.uniform(1.8, 2.8)
        rows.append((time, v, amount, card, 0))

    fraud_cards = rng.choice(n_cards, size=14, replace=False)
    created = 0
    while created < n_fraud:
        card = int(rng.choice(fraud_cards))
        day = int(rng.integers(0, 2))
        night = rng.random() < 0.65                  # 35% das fraudes em horario comercial
        hour = float(np.clip(rng.normal(3.2, 1.1), 0.05, 5.9)) if night \
            else float(np.clip(rng.normal(15.0, 4.0), 6.5, 23.5))
        base = day * 86400 + hour * 3600
        burst = int(min(rng.integers(1, 4), n_fraud - created))
        for step in range(burst):
            time = base + step * float(rng.uniform(18, 75))
            draw = rng.random()
            if draw < 0.22:
                amount = float(rng.uniform(1.0, 9.0))               # teste de cartao
            elif draw < 0.35:
                amount = float(np.clip(rng.lognormal(3.4, 0.9), 5, 400))  # valor comum: dificil
            else:
                amount = float(np.clip(rng.lognormal(6.1, 0.7), 300, 12000))
            v = rng.normal(0, 1, 28)
            if rng.random() > 0.06:                  # 6% das fraudes nao deslocam as PCA
                # 4 das 6 componentes se deslocam, com intensidade moderada
                picks = rng.choice([13, 16, 11, 9, 3, 10], size=4, replace=False)
                for column in picks:
                    direction = -1.0 if column in (13, 16, 11, 9) else 1.0
                    v[column] += direction * rng.uniform(1.8, 4.0)
            rows.append((time, v, amount, card, 1))
            created += 1

    rows.sort(key=lambda r: r[0])
    data = {"Time": [round(r[0], 2) for r in rows]}
    for j in range(28):
        data[f"V{j + 1}"] = [round(float(r[1][j]), 6) for r in rows]
    data["Amount"] = [round(r[2], 2) for r in rows]
    data["card_id"] = [f"CARD{r[3]:03d}" for r in rows]
    data["Class"] = [r[4] for r in rows]
    frame = pd.DataFrame(data)
    frame.insert(0, "tx_id", [4000 + i for i in range(len(frame))])
    return frame


# --------------------------------------------------------------------------- #
# 2. E-mails (Enron-like legitimo + phishing BR)
# --------------------------------------------------------------------------- #
LEGIT_SUBJECTS = [
    "Ata da reuniao de {dia}", "Relatorio semanal de vendas - semana {n}",
    "Proposta comercial {empresa}", "Convite: alinhamento do projeto {proj}",
    "Nota fiscal {n} referente ao servico de consultoria",
    "Ferias aprovadas - {pessoa}", "Pauta do comite de {area}",
    "Onboarding do novo analista de {area}", "Follow-up da call de ontem",
    "Contrato {n} para assinatura", "Fechamento contabil de {mes}",
    "Resultado do teste de carga no ambiente de homologacao",
    "Atualizacao do cronograma do projeto {proj}", "Feedback da sprint {n}",
    "Reserva de sala para a apresentacao de {mes}",
    "Urgente: cliente {empresa} pediu retorno hoje",
    "Verifique agora os numeros da planilha de {area}",
    "Segunda via do boleto do fornecedor {empresa}",
    "Pagamento via pix ao fornecedor - confirmacao",
]
LEGIT_BODIES = [
    "Ola {pessoa},\n\nSegue o resumo do que combinamos na reuniao e os proximos passos do projeto {proj}. "
    "Qualquer ajuste me avise ate {dia}.\n\nAbraco,\n{remetente}\n{empresa}",
    "Prezados,\n\nEncaminho o relatorio da semana {n} com os numeros de {area}. Os detalhes estao na planilha anexa. "
    "Podemos discutir na reuniao de {dia}.\n\nAtenciosamente,\n{remetente}",
    "Oi {pessoa},\n\nConforme conversamos, a proposta para {empresa} esta anexada. O prazo de validade e de 30 dias e "
    "o escopo cobre as tres frentes que voce pediu.\n\nObrigado,\n{remetente}",
    "Time,\n\nO deploy em homologacao foi concluido. O ambiente esta disponivel em https://homolog.{dominio}/painel "
    "para validacao funcional. Reportem os problemas pelo Jira.\n\n{remetente}",
    "{pessoa}, tudo bem?\n\nRevisei o documento e deixei comentarios nas secoes 2 e 4. A versao consolidada esta em "
    "https://docs.google.com/document/d/{token}\n\nAbraco,\n{remetente}",
    "Bom dia,\n\nSegue a nota fiscal {n} do servico de consultoria prestado em {mes}. O pagamento segue o combinado, "
    "boleto com vencimento em 15 dias.\n\nFinanceiro - {empresa}",
    "Pessoal,\n\nA pauta do comite de {area} esta definida: orcamento, contratacoes e o roadmap do projeto {proj}. "
    "Chamada no calendario para {dia}.\n\n{remetente}",
    "Hi {pessoa},\n\nPlease find attached the updated forecast for {area}. The numbers were reviewed with the "
    "regional team. Let us discuss on {dia}.\n\nBest regards,\n{remetente}",
    "{pessoa},\n\nO cliente pediu retorno urgente sobre a proposta. Verifique agora se a versao no portal esta correta: "
    "https://{dominio}/propostas/{n}\n\nObrigado,\n{remetente}",
    "Oi,\n\nSegue a segunda via do boleto do fornecedor. O pagamento pode ser feito via pix pela conta da empresa. "
    "O comprovante de pagamento esta em anexo.\n\n{remetente} - {empresa}",
]
PHISH_SUBJECTS = [
    "URGENTE: sua conta {brand} sera bloqueada em 24 horas",
    "[{brand}] Confirme seus dados agora para evitar o bloqueio",
    "Restituicao do IRPF liberada - resgate seu valor hoje",
    "Sua encomenda esta retida - pague a taxa de liberacao",
    "Pix nao autorizado detectado na sua conta {brand}",
    "Seu CPF foi negativado - regularize em 24 horas",
    "ULTIMO AVISO: fatura em atraso do {brand}",
    "{brand}: token de seguranca expirado, atualize seu cadastro",
    "Voce foi selecionado! Premio de R$ 5.000 do {brand}",
    "Comprovante de pagamento em anexo - confira agora",
    "Assinatura {brand} suspensa: atualize a forma de pagamento",
    "Processo judicial em seu nome - intimacao eletronica",
]
PHISH_BODIES = [
    "Prezado cliente,\n\nDetectamos uma movimentacao suspeita e sua conta sera bloqueada. "
    "Clique aqui para verificar agora: {url}\n\nVoce tem 24 horas para confirmar seus dados, caso contrario a conta "
    "sera suspensa definitivamente.\n\nAtenciosamente,\nCentral de Seguranca {brand}",
    "Sua fatura esta em atraso e o nome sera negativado.\n\nAcesse {url} e emita a segunda via do boleto "
    "para evitar restricao no seu CPF.\n\nSetor de Cobranca {brand}",
    "Informamos que sua restituicao foi liberada. Para receber o valor via pix, confirme seus dados bancarios em {url}\n\n"
    "O prazo final para resgate expira hoje.\n\nReceita Federal - {brand}",
    "Sua encomenda esta retida na alfandega por pendencia de taxa. Regularize em {url} para liberar a entrega.\n\n"
    "Nao ignore este ultimo aviso.\n\n{brand}",
    "Detectamos um pix nao autorizado de R$ 4.780,00 na sua conta. Se nao reconhece, cancele imediatamente: {url}\n\n"
    "Informe sua senha de 6 digitos para validar o cancelamento.\n\nSeguranca {brand}",
    "Parabens! Voce foi selecionado no sorteio {brand} e tem um premio de R$ 5.000 para resgatar. "
    "Clique aqui: {url}\n\nVagas limitadas, resgate seu premio em 24 horas.",
    "Segue em anexo o comprovante de pagamento. Abra o documento e confirme os dados em {url} para dar baixa na "
    "pendencia.\n\nDepartamento Financeiro {brand}",
    "Sua assinatura foi suspensa por falha na cobranca. Atualize seu cadastro em {url} para reativar o acesso "
    "agora mesmo.\n\nEquipe {brand}",
]
# BEC / conta legitima comprometida: texto curto, sem link, sem palavra-gatilho.
# Sao os casos que o baseline textual nao pega - e por isso o recall nunca da 1.0.
LOW_SIGNAL_PHISH = [
    ("Confirmacao de dados bancarios do fornecedor",
     "Bom dia,\n\nMudamos a conta de recebimento. Pode atualizar o cadastro do fornecedor para a nova conta "
     "antes do pagamento de amanha? Envio os dados no retorno deste e-mail.\n\nObrigado,\n{remetente}"),
    ("Sobre o pagamento de ontem",
     "Oi {pessoa},\n\nO pagamento de ontem nao caiu na nossa conta. Consegue verificar com o financeiro e me "
     "responder ainda hoje? Estou em reuniao, prefiro por e-mail.\n\n{remetente}"),
    ("Documento para revisao",
     "{pessoa}, segue o documento que comentamos na call. Preciso da sua validacao para seguir com o contrato.\n\n"
     "Abraco,\n{remetente}"),
    ("Alteracao de dados cadastrais",
     "Prezados,\n\nSolicito a alteracao do e-mail de contato da nossa conta corporativa. O novo endereco sera usado "
     "para as notas fiscais a partir deste mes.\n\nAtenciosamente,\n{remetente}"),
]
BRANDS = [
    ("Itau", "itau", "itau.com.br"), ("Nubank", "nubank", "nubank.com.br"),
    ("Bradesco", "bradesco", "bradesco.com.br"), ("Caixa", "caixa", "caixa.gov.br"),
    ("Correios", "correios", "correios.com.br"), ("Netflix", "netflix", "netflix.com"),
    ("Receita Federal", "receita", "gov.br"), ("Serasa", "serasa", "serasa.com.br"),
    ("PicPay", "picpay", "picpay.com"), ("Mercado Pago", "mercadopago", "mercadopago.com.br"),
    ("Santander", "santander", "santander.com.br"), ("Magalu", "magalu", "magazineluiza.com.br"),
]
PEOPLE = ["Ana", "Bruno", "Carla", "Diego", "Eduardo", "Fernanda", "Gustavo", "Helena",
          "Igor", "Juliana", "Lucas", "Mariana", "Nara", "Otavio", "Paula", "Rafael"]
COMPANIES = ["Vega Log", "Atlas Energia", "Norte Seguros", "Prisma Varejo", "Delta Saude",
             "Orion Telecom", "Sigma Alimentos", "Rio Branco Industria"]
AREAS = ["operacoes", "financeiro", "tecnologia", "vendas", "juridico", "suprimentos"]
PROJECTS = ["Aurora", "Baobab", "Cobalto", "Dunas", "Everest", "Farol"]
MONTHS = ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto"]
DAYS = ["segunda", "terca", "quarta", "quinta", "sexta"]
TYPO_TEMPLATES = ["{b}-seguro.top", "{b}-atendimento.xyz", "seguranca-{b}.click", "{b}.verifica-cliente.com",
                  "central{b}.icu", "{b}online.zip", "meu{b}.work", "{b}-clientes.sbs"]
SHORTENERS = ["bit.ly", "cutt.ly", "encurtador.com.br", "tinyurl.com", "is.gd"]


def _token(rng, size=14):
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(list(chars), size=size))


def gen_emails(n_legit: int = 760, n_phish: int = 210) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    rows = []

    for i in range(n_legit):
        person = str(rng.choice(PEOPLE))
        sender = str(rng.choice(PEOPLE)).lower()
        company = str(rng.choice(COMPANIES))
        domain = company.lower().replace(" ", "") + ".com.br"
        fields = dict(
            dia=str(rng.choice(DAYS)), n=int(rng.integers(100, 999)), empresa=company,
            proj=str(rng.choice(PROJECTS)), pessoa=person, area=str(rng.choice(AREAS)),
            mes=str(rng.choice(MONTHS)), remetente=sender.capitalize(), dominio=domain,
            token=_token(rng, 22),
        )
        subject = str(rng.choice(LEGIT_SUBJECTS)).format(**fields)
        body = str(rng.choice(LEGIT_BODIES)).format(**fields)

        # ruido realista do lado legitimo
        if rng.random() < 0.10:
            body += f"\n\nLink rapido: https://{rng.choice(SHORTENERS)}/{_token(rng, 6)}"
        if rng.random() < 0.08:
            brand = str(rng.choice(BRANDS)[2])
            body += f"\n\nCanal oficial do banco: https://{brand}/empresas"
        spf = "pass" if rng.random() > 0.12 else str(rng.choice(["softfail", "none"]))
        dkim = "pass" if rng.random() > 0.10 else str(rng.choice(["none", "fail"]))
        attachment = str(rng.choice(["", "relatorio.pdf", "planilha.xlsx", "contrato.docx",
                                     "ata.pdf", "backup.zip"]))
        rows.append({
            "message_id": f"legit-{i:04d}@{domain}",
            "from_name": f"{sender.capitalize()} {rng.choice(['Silva', 'Souza', 'Almeida', 'Costa'])}",
            "from_addr": f"{sender}.{rng.choice(['silva', 'souza', 'costa'])}@{domain}",
            "reply_to": "" if rng.random() < 0.9 else f"{sender}@{rng.choice(COMPANIES)[:4].lower()}parceiro.com.br",
            "subject": subject,
            "body": body,
            "spf_result": spf,
            "dkim_result": dkim,
            "domain_age_days": int(rng.integers(700, 7000)) if rng.random() > 0.05 else int(rng.integers(40, 200)),
            "attachments": attachment,
            "urls": "",
            "label": 0,
        })

    n_low_signal = int(n_phish * 0.22)
    for i in range(n_phish):
        brand_name, brand_key, official = [str(x) for x in rng.choice(BRANDS)]

        if i < n_low_signal:
            # BEC (conta corporativa comprometida): dominio legitimo, SPF/DKIM validos,
            # texto identico ao de um e-mail comum de trabalho. Indistinguivel para o
            # baseline textual - e exatamente por isso o recall nunca chega a 1.0.
            person = str(rng.choice(PEOPLE))
            sender = str(rng.choice(PEOPLE)).lower()
            company = str(rng.choice(COMPANIES))
            domain = company.lower().replace(" ", "") + ".com.br"
            fields = dict(
                dia=str(rng.choice(DAYS)), n=int(rng.integers(100, 999)), empresa=company,
                proj=str(rng.choice(PROJECTS)), pessoa=person, area=str(rng.choice(AREAS)),
                mes=str(rng.choice(MONTHS)), remetente=sender.capitalize(), dominio=domain,
                token=_token(rng, 22),
            )
            if rng.random() < 0.5:
                subject = str(rng.choice(LEGIT_SUBJECTS)).format(**fields)
                body = str(rng.choice(LEGIT_BODIES)).format(**fields)
            else:
                pair = LOW_SIGNAL_PHISH[int(rng.integers(0, len(LOW_SIGNAL_PHISH)))]
                subject, body = str(pair[0]), str(pair[1]).format(**fields)
            rows.append({
                "message_id": f"phish-bec-{i:04d}@{domain}",
                "from_name": f"{sender.capitalize()} {rng.choice(['Silva', 'Souza', 'Almeida', 'Costa'])}",
                "from_addr": f"{sender}.{rng.choice(['silva', 'souza', 'costa'])}@{domain}",
                "reply_to": "" if rng.random() < 0.8 else f"{sender}.financeiro@{domain}",
                "subject": subject,
                "body": body,
                "spf_result": "pass",
                "dkim_result": "pass",
                "domain_age_days": int(rng.integers(700, 7000)),
                "attachments": str(rng.choice(["", "relatorio.pdf", "documento.pdf", "planilha.xlsx"])),
                "urls": "",
                "label": 1,
            })
            continue

        style = rng.random()
        if style < 0.45:
            host = str(rng.choice(TYPO_TEMPLATES)).format(b=brand_key)
            url = f"http://{host}/{rng.choice(['login', 'acesso', 'validar', 'atualizar'])}.php?id={_token(rng, 10)}"
        elif style < 0.62:
            url = (f"http://{rng.integers(1, 223)}.{rng.integers(0, 255)}."
                   f"{rng.integers(0, 255)}.{rng.integers(1, 254)}/{brand_key}/login")
            host = url.split("/")[2]
        elif style < 0.85:
            host = str(rng.choice(SHORTENERS))
            url = f"https://{host}/{_token(rng, 7)}"
        else:
            host = f"{brand_key}.{_token(rng, 8)}.com"
            url = f"https://{host}/seguranca/validacao"

        subject = str(rng.choice(PHISH_SUBJECTS)).format(brand=brand_name)
        body = str(rng.choice(PHISH_BODIES)).format(url=url, brand=brand_name)
        sender_domain = host if "." in host and not host[0].isdigit() \
            else str(rng.choice(TYPO_TEMPLATES)).format(b=brand_key)
        if rng.random() < 0.3:
            sender_domain = str(rng.choice(["gmail.com", "hotmail.com", "outlook.com"]))
        compromised = rng.random() < 0.22          # conta invadida: SPF/DKIM validos
        rows.append({
            "message_id": f"phish-{i:04d}@{sender_domain}",
            "from_name": f"{brand_name} {rng.choice(['Seguranca', 'Atendimento', 'Central', 'Cobranca'])}"
            if rng.random() < 0.7 else f"{rng.choice(PEOPLE)} {rng.choice(['Lima', 'Reis'])}",
            "from_addr": f"{rng.choice(['seguranca', 'atendimento', 'nao-responda', 'suporte'])}@{sender_domain}",
            "reply_to": "" if rng.random() < 0.55 else
            f"retorno{int(rng.integers(10, 99))}@{rng.choice(['gmail.com', 'mail.ru', 'yandex.com'])}",
            "subject": subject,
            "body": body,
            "spf_result": "pass" if compromised else str(rng.choice(["fail", "softfail", "none"])),
            "dkim_result": "pass" if compromised else str(rng.choice(["fail", "none"])),
            "domain_age_days": int(rng.integers(600, 4000)) if compromised else int(rng.integers(1, 90)),
            "attachments": str(rng.choice(["", "", "comprovante.zip", "fatura.html",
                                           "documento.exe", "nota.pdf.zip"])),
            "urls": url,
            "label": 1,
        })

    return pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Redes sociais (perfis autenticos x bots/golpes)
# --------------------------------------------------------------------------- #
BOT_POSTS = [
    "RENDA EXTRA de R$ 300 por dia!!! clique no link da bio e comece hoje {url}",
    "GANHE DINHEIRO em casa!!! vagas limitadas, chama no direct agora {url}",
    "Sinal gratis de aposta, lucro garantido hoje!!! {url}",
    "Airdrop de cripto liberado! dobre seu investimento em 24h {url}",
    "SEGUIDORES GRATIS!!! siga de volta e me chama no whats {url}",
    "Cupom exclusivo de R$ 200, ultimas vagas, clique no link {url}",
]
# bots bem feitos: texto humano, sem link, com bio e foto - o caso difícil
SUBTLE_BOT_POSTS = [
    "que dia lindo hoje, aproveitando o sol",
    "concordo demais com esse ponto, precisa ser dito",
    "melhor serie do ano sem discussao",
    "bom dia a todos, foco na meta da semana",
    "quem ai tambem acordou cedo pra treinar?",
]
HUMAN_POSTS = [
    "bom dia, cafe e mais um dia de trabalho por aqui",
    "alguem ai assistiu o jogo de ontem? que segundo tempo",
    "terminei de ler o livro que a {pessoa} indicou, valeu a pena",
    "chuva forte aqui em {cidade} hoje, transito parado",
    "consegui finalmente organizar a planilha do mes, vitoria pessoal",
    "recomendo muito esse restaurante novo no centro de {cidade}",
    "treino feito, 6km hoje. devagar mas chegando",
    "opiniao impopular: filme antigo e melhor que remake",
    "dica pra quem estuda seguranca: pratique em laboratorio proprio",
    "obrigada pelas mensagens de aniversario, gente",
]
# pequeno negocio real divulgando promocao: parece spam, mas e legitimo
PROMO_HUMAN_POSTS = [
    "promocao de hoje na loja: cupom de 20% pra quem chegar antes do meio dia",
    "vagas limitadas no curso de bolo caseiro, chama no direct pra reservar",
    "renda extra? venha ser revendedora comigo, explico tudo {url}",
]
TAILS = [
    "", " enfim", " no fim deu certo", " recomendo", " amanha continuo", " que dia",
    " bora pra cima", " no mais, tudo tranquilo", " depois conto o resto",
    " valeu quem ajudou", " ate semana que vem", " coisa de louco",
]
CITIES = ["Sao Paulo", "Recife", "Curitiba", "Salvador", "Belem", "Porto Alegre", "Goiania"]
BIOS = ["fotografa e mae de dois", "dev backend, cafe e gatos", "professora de historia",
        "corredora amadora | 42k", "analista de dados em {cidade}", "musico nos fins de semana",
        "estudante de enfermagem", "pai do Theo, torcedor sofredor"]


def gen_social(n_real: int = 2300, n_fake: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    rows = []

    for i in range(n_real):
        new_account = rng.random() < 0.08              # conta nova legitima
        days = float(rng.integers(3, 90)) if new_account else float(rng.integers(120, 5200))
        created = TODAY - timedelta(days=days, hours=float(rng.integers(0, 24)))
        age = max((TODAY - created).days, 1)
        followers = int(np.clip(rng.gamma(1.6, 8), 0, 120)) if new_account \
            else int(np.clip(rng.lognormal(5.4, 1.15), 15, 90000))
        following = int(np.clip(followers * rng.uniform(0.25, 2.2) + (400 if new_account else 0), 12, 5000))
        posts = int(np.clip(rng.lognormal(6.2, 1.1), 20, 60000)) if not new_account \
            else int(np.clip(rng.normal(age * 3, 20), 3, 900))
        city = str(rng.choice(CITIES))
        if rng.random() < 0.05:
            text = str(rng.choice(PROMO_HUMAN_POSTS)).format(
                url=f"https://{rng.choice(SHORTENERS)}/{_token(rng, 6)}")
        else:
            text = str(rng.choice(HUMAN_POSTS)).format(pessoa=str(rng.choice(PEOPLE)), cidade=city)
            text += str(rng.choice(TAILS))
            if rng.random() < 0.4:
                text += f" ({int(rng.integers(2, 99))})"
        suffix = str(int(rng.integers(0, 99))) if rng.random() < 0.5 else ""
        rows.append({
            "username": f"{str(rng.choice(PEOPLE)).lower()}_"
                        f"{str(rng.choice(['silva', 'souza', 'lima', 'rocha', 'alves']))}{suffix}",
            "name": f"{rng.choice(PEOPLE)} {rng.choice(['Silva', 'Souza', 'Lima', 'Rocha', 'Alves'])}",
            "bio": str(rng.choice(BIOS)).format(cidade=city) if rng.random() > 0.12 else "",
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "account_age_days": age,
            "followers": followers,
            "following": following,
            "statuses_count": posts,
            "default_profile_image": int(rng.random() < 0.06),
            "verified": int(rng.random() < 0.01),
            "post_text": text,
            "label": 0,
        })

    burst_days = [TODAY - timedelta(days=float(d)) for d in rng.integers(5, 400, size=9)]
    n_subtle = int(n_fake * 0.20)
    for i in range(n_fake):
        subtle = i < n_subtle
        if subtle:
            # bot maduro ("conta dormente" comprada): fora da rajada, com bio, foto,
            # razao de seguidores plausivel e post de aparencia humana
            created = TODAY - timedelta(days=float(rng.integers(200, 3000)),
                                        hours=float(rng.integers(0, 24)))
            age = max((TODAY - created).days, 1)
            followers = int(np.clip(rng.lognormal(5.2, 1.0), 30, 20000))
            following = int(np.clip(followers * rng.uniform(0.4, 2.1), 20, 5000))
            posts = int(np.clip(rng.lognormal(6.0, 1.0), 30, 40000))
            pool = SUBTLE_BOT_POSTS if rng.random() < 0.4 else HUMAN_POSTS
            text = str(rng.choice(pool)).format(pessoa=str(rng.choice(PEOPLE)),
                                                cidade=str(rng.choice(CITIES)))
            text += str(rng.choice(TAILS))
            bio = str(rng.choice(BIOS)).format(cidade=str(rng.choice(CITIES)))
            username = (f"{str(rng.choice(PEOPLE)).lower()}_"
                        f"{str(rng.choice(['silva', 'souza', 'lima', 'rocha']))}"
                        f"{int(rng.integers(1, 99)) if rng.random() < 0.5 else ''}")
            name = f"{rng.choice(PEOPLE)} {rng.choice(['Martins', 'Prado', 'Neves'])}"
            default_image = int(rng.random() < 0.08)
        else:
            base = burst_days[int(rng.integers(0, len(burst_days)))]
            created = base + timedelta(minutes=float(rng.integers(0, 260)))
            age = max((TODAY - created).days, 1)
            followers = int(np.clip(rng.gamma(1.4, 9), 0, 300))
            following = int(np.clip(rng.normal(1800, 600), 200, 5000))
            posts = int(np.clip(rng.normal(age * rng.uniform(8, 60), 60), 5, 40000))
            url = f"https://{rng.choice(SHORTENERS)}/{_token(rng, 6)}"
            text = str(rng.choice(BOT_POSTS)).format(url=url)
            bio = "" if rng.random() < 0.7 else f"link na bio {url}"
            username = (f"{rng.choice(['user', 'ana', 'promo', 'ganhe', 'lucro', 'invest'])}"
                        f"{int(rng.integers(100000, 9999999))}")
            name = str(rng.choice(["Renda Extra 24h", "Promo 999", "Invest Bot", "Ganhe 1000", "Ana 7712"]))
            default_image = int(rng.random() < 0.8)
        rows.append({
            "username": username,
            "name": name,
            "bio": bio,
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "account_age_days": age,
            "followers": followers,
            "following": following,
            "statuses_count": posts,
            "default_profile_image": default_image,
            "verified": 0,
            "post_text": text,
            "label": 1,
        })

    return pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Arquivos de entrada avulsos para o modo --predict
# --------------------------------------------------------------------------- #
PHISH_EML = """From: Itau Unibanco Seguranca <seguranca@itau-seguro.top>
Reply-To: retorno42@mail.ru
To: cliente@empresa.com.br
Subject: URGENTE: sua conta Itau sera bloqueada em 24 horas
Date: Thu, 18 Sep 2026 03:47:11 -0300
Message-ID: <a91f2@itau-seguro.top>
Authentication-Results: mx.empresa.com.br; spf=fail smtp.mailfrom=itau-seguro.top; dkim=fail
Content-Type: text/plain; charset="utf-8"

Prezado cliente,

Detectamos uma movimentacao suspeita e sua conta sera bloqueada.
Clique aqui para verificar agora: http://itau.verifica-cliente.top/login.php?id=8fa20b1c

Voce tem 24 horas para confirmar seus dados e informe sua senha de 6 digitos
para validar o cancelamento do pix nao autorizado de R$ 4.780,00.

Atenciosamente,
Central de Seguranca Itau
"""

LEGIT_EML = """From: Mariana Costa <mariana.costa@vegalog.com.br>
To: time@vegalog.com.br
Subject: Ata da reuniao de quinta - projeto Aurora
Date: Thu, 18 Sep 2026 14:12:05 -0300
Message-ID: <77c11@vegalog.com.br>
Authentication-Results: mx.vegalog.com.br; spf=pass smtp.mailfrom=vegalog.com.br; dkim=pass
Content-Type: text/plain; charset="utf-8"

Ola time,

Segue o resumo do que combinamos na reuniao e os proximos passos do projeto Aurora.
A versao consolidada do documento esta em https://docs.google.com/document/d/8812kkqa01

Qualquer ajuste me avise ate sexta.

Abraco,
Mariana
Vega Log
"""


def write_side_inputs(transactions: pd.DataFrame, social: pd.DataFrame) -> None:
    (OUT / "exemplo_phishing.eml").write_text(PHISH_EML, encoding="utf-8")
    (OUT / "exemplo_legitimo.eml").write_text(LEGIT_EML, encoding="utf-8")

    fraudulent = transactions[transactions.Class == 1].head(3).drop(columns=["Class"])
    fraudulent.to_json(OUT / "transacoes_suspeitas.json", orient="records", indent=2)

    bots = social[social.label == 1].head(3).drop(columns=["label"])
    bots.to_json(OUT / "perfis_suspeitos.json", orient="records", indent=2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    transactions = gen_transactions()
    emails = gen_emails()
    social = gen_social()

    transactions.to_csv(OUT / "transacoes_amostra.csv", index=False)
    emails.to_csv(OUT / "emails_amostra.csv", index=False)
    social.to_csv(OUT / "social_amostra.csv", index=False)
    write_side_inputs(transactions, social)

    print(f"transacoes : {len(transactions):>6} linhas | fraudes {int(transactions.Class.sum())} "
          f"({transactions.Class.mean() * 100:.2f}%)")
    print(f"emails     : {len(emails):>6} linhas | phishing {int(emails.label.sum())} "
          f"({emails.label.mean() * 100:.2f}%)")
    print(f"social     : {len(social):>6} linhas | fraudulentos {int(social.label.sum())} "
          f"({social.label.mean() * 100:.2f}%)")
    print(f"amostras em: {OUT}")


if __name__ == "__main__":
    main()
