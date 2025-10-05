# Makefile para K-means 1D Sequencial - Etapa 0

CC = gcc
CFLAGS = -O2 -std=c99 -lm
PROG = kmeans_1d_naive.exe
FONTE = kmeans_1d_naive.c

RESULTS_DIR = resultados

$(PROG): $(FONTE)
	$(CC) -o $(PROG) $(FONTE) $(CFLAGS)
	@echo " >> Compilado: $(PROG)"

$(RESULTS_DIR):
	@if not exist $(RESULTS_DIR) mkdir $(RESULTS_DIR)

dados_pequenos: $(RESULTS_DIR)
	@echo " >> gerando dados pequenos (N=10k, K=4)..."
	python scripts/gerar_dados.py 10000 4 $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv

dados_medios: $(RESULTS_DIR)
	@echo " >> gerando dados medios (N=100k, K=8)..."
	python scripts/gerar_dados.py 100000 8 $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv

dados_grandes: $(RESULTS_DIR)
	@echo " >> gerando dados grandes (N=1M, K=16)..."
	python scripts/gerar_dados.py 1000000 16 $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv

dados_todos: dados_pequenos dados_medios dados_grandes
	@echo " >> todos os dados de teste gerados"

#teste dados no roteiro
teste_original: $(PROG) $(RESULTS_DIR)
	@echo " >> exe teste original com dados_iniciais.csv e centroides_iniciais.csv..."
	$(PROG) dados_iniciais.csv centroides_iniciais.csv 100 1e-6 assign_iniciais.csv centros_iniciais.csv resultados_desempenho.csv

teste_pequeno: $(PROG) dados_pequenos
	@echo " >> exe teste pequeno (N=10k, K=4)..."
	$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/assign_10k_4.csv $(RESULTS_DIR)/centros_10k_4.csv $(RESULTS_DIR)/resultados_desempenho.csv

teste_medio: $(PROG) dados_medios
	@echo " >> exe teste medio (N=100k, K=8)..."
	$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_100k_8.csv $(RESULTS_DIR)/centros_100k_8.csv $(RESULTS_DIR)/resultados_desempenho.csv

teste_grande: $(PROG) dados_grandes
	@echo " >> exe teste grande (N=1M, K=16)..."
	$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/assign_1M_16.csv $(RESULTS_DIR)/centros_1M_16.csv $(RESULTS_DIR)/resultados_desempenho.csv

teste_todos: teste_pequeno teste_medio teste_grande
	@echo " >> todos os testes executados"

graficos: 
	@echo " >> gerando graficos a partir dos resultados..."
	python scripts/gerar_graficos.py

relatorio: $(PROG) dados_todos teste_todos graficos
	@echo " >> relatorio completo gerado"

clean:
	@if exist $(PROG) del /Q $(PROG)
	@if exist $(RESULTS_DIR) rmdir /S /Q $(RESULTS_DIR)
	@echo " >> Arquivos gerados removidos"

.PHONY: clean teste_todos dados_todos graficos relatorio
