# README - Projeto PCD: K-means 1D 

Implementação do algoritmo de k-means implementação serial, pararelização com OpenMP, CUDA e MPI (uitlizando a implementação OpenMPI)

## Criação de um ambiente virtual Python
Utilizamos Python no desenvolvimento do trabalho para geração de gráficos e dados para execução das diferentes baterias de teste.
Para criar um ambiente virtual python, junto com as dependências necessárias, execute os seguintes comandos na raiz do projeto:
```bash
python3 -m venv venv
pip install -r requirements.txt
```

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

- MPI:
```bash
cd mpi
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
