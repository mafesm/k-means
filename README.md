# README - Projeto PCD: K-means 1D - Versão Sequencial (Etapa 0)

## Descrição

Esta pasta implementa a **Etapa 0** do trabalho de K-means 1D, com a versão sequencial do algoritmo que servirá como **baseline** para comparação com as implementações paralelizadas nas próximas etapas.

## Estrutura

```
etapa_0/
│
├── Makefile                          # Script de compilação
├── kmeans_1d_naive.c                 # Implementação sequencial
│
├── scripts/
│   ├── gerar_dados.py                # Geração de dados de teste
│   └── gerar_graficos.py             # Geração de gráficos a partir dos resultados
│
└── resultados/                       # Pasta de saída
    ├── dados_10k.csv                 # Dados pequenos (10.000 pontos, 4 clusters)
    ├── centroides_4.csv              # Centróides iniciais para teste pequeno
    ├── assign_10k_4.csv              # Atribuições de clusters para teste pequeno
    ├── centros_10k_4.csv             # Centróides finais para teste pequeno
    [...]                             # Demais testes
    ├── tempos_execucao.png           # Gráfico de tempos de execução
    ├── sse_final.png                 # Gráfico de SSE final
    ├── iteracoes.png                 # Gráfico de iterações
    └── resumo_resultados.txt         # Resumo dos resultados
```

## Compilação

```bash
# Executa todo o fluxo
make relatorio
```

### Outros Passos:
```bash
# 1. Compilar
make

# 2. Gerar dados de teste
make dados_todos

# 3. Executar testes (resultados salvos automaticamente)
make teste_todos

# 4. Ver resultados coletados
make resultados

# 5. Gerar gráficos
make graficos
```

## Bateria de Testes

| Teste   | (N)        | (K)  |
|---------|------------|------|
| Pequeno | 10.000     | 4    |
| Médio   | 100.000    | 8    |
| Grande  | 1.000.000  | 16   |


## Autores

Projeto desenvolvido para a disciplina de **Programação Concorrente e Distribuída (PCD)** 

- 
- Maria Fernanda Siqueira de Moraes
- 

## Licença

Este projeto é para fins educacionais.

---
