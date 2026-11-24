# Trabalho Prático - Otimização 1 (CEFET-MG)

**Professor:** Flávio Vinícius Cruzeiro Martins  
**Tema:** Posicionamento e Cobertura em Redes de Sensores Sem Fio (RSSF)

---

## 📌 Sobre o Problema

O objetivo do trabalho é encontrar as melhores posições para instalar sensores em uma área geográfica, de modo a:
1. **Cobrir 100% dos pontos da região** (com o raio de sensoriamento de cada nó).
2. **Garantir conectividade** (todos os sensores instalados devem conseguir transmitir seus dados via rádio até uma estação central / *Sink*).
3. **Minimizar o custo total de instalação.**

---

## ⚙️ O que tem no código (`experimento_rssf.py`)

O script compara duas abordagens para resolver o mesmo problema:

1. **Modelo Exato - PLI (Programação Linear Inteira):**
   * Baseado na modelagem de fluxo do artigo de **Fonseca et al. (SBPO 2019 / CEFET-MG)**.
   * Utiliza o solver CBC (via biblioteca `PuLP`) para encontrar a **solução ótima comprovada** (gabarito).
2. **Meta-heurística - GRASP:**
   * Atende à sugestão do professor no enunciado.
   * Executa em duas fases (Construção Gulosa Aleatorizada com lista de candidatos $\alpha = 0.3$ + Busca Local de remoção de redundâncias).
   * Encontra soluções quase idênticas ao ótimo em milissegundos.

---

## 🚀 Como Executar

### 1. Instalar as dependências
```bash
pip install pulp numpy matplotlib
```

### 2. Rodar o experimento
```bash
python experimento_rssf.py
```

---

## 📊 Arquivos Gerados

Ao rodar o script, ele testa instâncias de tamanhos variados ($N = 10$ a $120$ nós) e gera automaticamente:

* **`resultados_experimentos.csv`**: Tabela com os tempos de execução, custos e o GAP (%) de cada instância. Dá pra abrir no Excel e copiar direto para a tabela do nosso artigo.
* **`grafico_comparativo.png`**: Imagem com dois gráficos prontos para o artigo:
  1. Comparativo de tempo computacional (escala logarítmica).
  2. Gráfico de GAP (%) mostrando a qualidade da solução do GRASP em relação ao ótimo do Solver.
