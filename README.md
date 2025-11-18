# README - Projeto PCD: K-means 1D 

No momento, foram implementadas as três primeiras etapas do projeto: implementação serial, pararelização com OpenMP e CUDA.

## Compilação

Para cada versão do algoritmo, é necessário fazer a compilação em sua respectiva pasta:

- Serial:
```bash
cd serial
```

- OpenMP:
```bash
cd openmp
```

- CUDA:
```bash
cd cuda
```
Posteriormente, a compilação segue igualmente utlizando as diretivas abaixo: 

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

- Dante Y. Tsubono (163667)
- Maria Fernanda S. Moraes (165548)
- Wilson Cazarré (150452)

---
