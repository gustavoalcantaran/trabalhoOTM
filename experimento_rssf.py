import time
import math
import random
import csv
import numpy as np
import pulp
import matplotlib.pyplot as plt

# Trabalho Pratico - Otimizacao I (CEFET-MG)
# Prof. Flavio Martins
# Problema: Posicionamento e Cobertura em Redes de Sensores Sem Fio (RSSF)
# Modelo PLI baseado em Fonseca et al. (SBPO 2019 / CEFET-MG)
# Meta-heuristica: GRASP (Feo & Resende, 1989 / Silva et al., SBPO 2019)


def gerar_instancia(n, lado=100.0, r_sensor=35.0, r_com=45.0, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    pontos = []
    for _ in range(n):
        x = round(random.uniform(5.0, lado - 5.0), 2)
        y = round(random.uniform(5.0, lado - 5.0), 2)
        pontos.append((x, y))

    sink = (lado / 2.0, lado / 2.0)
    custos = [round(random.uniform(100.0, 250.0), 1) for _ in range(n)]

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = math.hypot(pontos[i][0] - pontos[j][0], pontos[i][1] - pontos[j][1])

    dist_sink = [math.hypot(p[0] - sink[0], p[1] - sink[1]) for p in pontos]

    cob = np.zeros((n, n), dtype=int)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if dist[i, j] <= r_sensor:
                cob[i, j] = 1
            if i != j and dist[i, j] <= r_com:
                adj[i, j] = 1

    return {
        'n': n,
        'pontos': pontos,
        'sink': sink,
        'custos': custos,
        'dist': dist,
        'dist_sink': dist_sink,
        'cob': cob,
        'adj': adj,
        'rc': r_com
    }


def resolver_pli(inst, timeout=60):
    n = inst['n']
    custos = inst['custos']
    cob = inst['cob']
    adj = inst['adj']
    dist_sink = inst['dist_sink']
    rc = inst['rc']

    prob = pulp.LpProblem("RSSF_PLI", pulp.LpMinimize)

    # x[i] = 1 se instala sensor no ponto i
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(n)}
    
    # f[i, j] = fluxo de comunicacao entre nos vizinhos
    arcos = [(i, j) for i in range(n) for j in range(n) if adj[i, j] == 1]
    f = {(i, j): pulp.LpVariable(f"f_{i}_{j}", lowBound=0, upBound=n, cat=pulp.LpContinuous) for (i, j) in arcos}
    f_sink = {i: pulp.LpVariable(f"fs_{i}", lowBound=0, upBound=n, cat=pulp.LpContinuous) for i in range(n)}

    # FO: minimizar custo total
    prob += pulp.lpSum([custos[i] * x[i] for i in range(n)])

    # Cobertura: todo ponto alvo j deve ser coberto por pelo menos 1 sensor
    for j in range(n):
        prob += pulp.lpSum([cob[i, j] * x[i] for i in range(n)]) >= 1

    # Conectividade: conservacao de fluxo em direcao ao sink
    for i in range(n):
        out_viz = pulp.lpSum([f[i, j] for j in range(n) if adj[i, j] == 1])
        out_sink = f_sink[i] if dist_sink[i] <= rc * 1.5 else 0
        in_viz = pulp.lpSum([f[j, i] for j in range(n) if adj[j, i] == 1])
        prob += (out_viz + out_sink - in_viz) == x[i]

    # Ativacao de arcos
    for (i, j) in arcos:
        prob += f[i, j] <= n * x[j]

    for i in range(n):
        if dist_sink[i] <= rc * 1.5:
            prob += f_sink[i] <= n * x[i]

    t0 = time.time()
    solver = pulp.PULP_CBC_CMD(timeLimit=timeout, msg=False)
    status_code = prob.solve(solver)
    tempo = time.time() - t0

    status = pulp.LpStatus[status_code]
    custo = None
    selecionados = []

    if prob.objective is not None and prob.objective.value() is not None:
        custo = round(pulp.value(prob.objective), 2)
        selecionados = [i for i in range(n) if pulp.value(x[i]) and pulp.value(x[i]) > 0.5]

    return {
        'status': status,
        'custo': custo,
        'tempo': round(tempo, 4),
        'sensores': selecionados
    }


def busca_local(sol, inst):
    """Refinamento: remove sensores cuja retirada mantem cobertura completa"""
    n = inst['n']
    cob = inst['cob']
    sol_refinada = set(sol)

    for u in list(sol_refinada):
        resto = sol_refinada - {u}
        cobertura = set()
        for v in resto:
            cobertura.update(np.where(cob[v] == 1)[0])
        if len(cobertura) == n:
            sol_refinada.remove(u)

    return sol_refinada


def construcao_grasp(inst, alpha=0.3):
    """Construcao gulosa aleatorizada com lista restrita de candidatos (LRC)"""
    n = inst['n']
    custos = inst['custos']
    cob = inst['cob']
    dist_sink = inst['dist_sink']
    rc = inst['rc']

    descobertos = set(range(n))
    sol = set()

    while descobertos:
        candidatos = []
        for i in range(n):
            if i in sol:
                continue
            cobertos = set(np.where(cob[i] == 1)[0])
            ganho = len(cobertos & descobertos)
            if ganho > 0:
                ratio = ganho / custos[i]
                candidatos.append((i, ratio))

        if not candidatos:
            restantes = [i for i in range(n) if i not in sol]
            sol.add(min(restantes, key=lambda k: custos[k]))
            break

        candidatos.sort(key=lambda item: item[1], reverse=True)
        r_max = candidatos[0][1]
        r_min = candidatos[-1][1]
        limite = r_max - alpha * (r_max - r_min)

        lrc = [item[0] for item in candidatos if item[1] >= limite]
        escolhido = random.choice(lrc)

        sol.add(escolhido)
        descobertos -= set(np.where(cob[escolhido] == 1)[0])

    # Garantir enlace com o sink
    if not any(dist_sink[i] <= rc * 1.5 for i in sol):
        ponte = min(range(n), key=lambda k: dist_sink[k])
        sol.add(ponte)

    return sol


def resolver_grasp(inst, max_iter=25, alpha=0.3):
    custos = inst['custos']
    t0 = time.time()

    melhor_sol = None
    melhor_custo = float('inf')

    for _ in range(max_iter):
        sol_inicial = construcao_grasp(inst, alpha=alpha)
        sol_ref = busca_local(sol_inicial, inst)
        custo = sum(custos[i] for i in sol_ref)

        if custo < melhor_custo:
            melhor_custo = custo
            melhor_sol = sol_ref

    tempo = time.time() - t0
    return {
        'custo': round(melhor_custo, 2),
        'tempo': round(tempo, 4),
        'sensores': list(melhor_sol) if melhor_sol else []
    }


def rodar_bateria_testes():
    # Instancias de pequeno a grande porte (para testar o limite do solver)
    tamanhos = [10, 15, 20, 30, 45, 60, 80, 100, 120]
    dados = []

    print(f"{'Instancia':<9} | {'N':<3} | {'Status PLI':<10} | {'T_PLI (s)':<10} | {'C_PLI':<9} | {'T_GRASP (s)':<11} | {'C_GRASP':<9} | {'GAP (%)':<8}")
    print("-" * 88)

    for n in tamanhos:
        inst = gerar_instancia(n, seed=42 + n)

        sol_pli = resolver_pli(inst, timeout=60)
        sol_grasp = resolver_grasp(inst, max_iter=25, alpha=0.3)

        c_pli = sol_pli['custo']
        c_gr = sol_grasp['custo']

        if c_pli and c_pli > 0:
            gap = ((c_gr - c_pli) / c_pli) * 100.0
            gap_txt = f"{gap:+.2f}%"
        else:
            gap = None
            gap_txt = "N/A"

        nome = f"rssf_{n:03d}"
        print(f"{nome:<9} | {n:<3} | {sol_pli['status']:<10} | {sol_pli['tempo']:>9.4f}s | {str(c_pli):>9} | {sol_grasp['tempo']:>10.4f}s | {c_gr:>9.2f} | {gap_txt:>8}")

        dados.append({
            'instancia': nome,
            'n': n,
            'status_pli': sol_pli['status'],
            't_pli': sol_pli['tempo'],
            'c_pli': c_pli,
            't_grasp': sol_grasp['tempo'],
            'c_grasp': c_gr,
            'gap': gap
        })

    salvar_csv(dados)
    salvar_graficos(dados)
    return dados


def salvar_csv(dados, arquivo="resultados_experimentos.csv"):
    campos = ['instancia', 'n', 'status_pli', 't_pli', 'c_pli', 't_grasp', 'c_grasp', 'gap']
    with open(arquivo, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(dados)
    print(f"\nTabela exportada em: {arquivo}")


def salvar_graficos(dados):
    ns = [d['n'] for d in dados]
    t_pli = [d['t_pli'] for d in dados]
    t_grasp = [d['t_grasp'] for d in dados]
    gaps = [d['gap'] for d in dados if d['gap'] is not None]
    ns_gap = [d['n'] for d in dados if d['gap'] is not None]

    plt.figure(figsize=(10, 4.5))

    # Grafico 1: Tempo Computacional (Escala Log)
    plt.subplot(1, 2, 1)
    plt.plot(ns, t_pli, 'r-o', label='PLI (Solver CBC)')
    plt.plot(ns, t_grasp, 'b--s', label='GRASP')
    plt.yscale('log')
    plt.xlabel('Tamanho da Instancia (N)')
    plt.ylabel('Tempo (s) [Escala Log]')
    plt.title('Comparativo de Tempo de Execucao')
    plt.grid(True, which="both", ls=":")
    plt.legend()

    # Grafico 2: GAP da Meta-heuristica vs Otimo
    plt.subplot(1, 2, 2)
    plt.plot(ns_gap, gaps, 'g-^', label='GAP GRASP vs Solver')
    plt.axhline(0, color='gray', linestyle=':')
    plt.xlabel('Tamanho da Instancia (N)')
    plt.ylabel('GAP (%)')
    plt.title('Qualidade das Solucoes (GAP %)')
    plt.grid(True, ls=":")
    plt.legend()

    plt.tight_layout()
    plt.savefig('grafico_comparativo.png', dpi=300)
    print("Graficos salvos em: grafico_comparativo.png")


if __name__ == '__main__':
    rodar_bateria_testes()
