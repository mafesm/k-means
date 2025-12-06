# Makefile para K-means 1D MPI - Etapa 3

MPICC = mpicc
CFLAGS = -O2 -std=c99 -lm
PROG = kmeans_1d_mpi
FONTE = kmeans_1d_mpi.c

RESULTS_DIR = resultados


$(PROG): $(FONTE)
	$(MPICC) $(CFLAGS) -o $(PROG) $(FONTE)
	@echo ">> Compilado: $(PROG)"

$(RESULTS_DIR):
	@mkdir -p $(RESULTS_DIR)

dados_pequenos: $(RESULTS_DIR)
	@echo ">> Gerando dados pequenos (N=10k, K=4)..."
	python3 ./scripts/gerar_dados.py 10000 4 $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv

dados_medios: $(RESULTS_DIR)
	@echo ">> Gerando dados medios (N=100k, K=8)..."
	python3 ./scripts/gerar_dados.py 100000 8 $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv

dados_grandes: $(RESULTS_DIR)
	@echo ">> Gerando dados grandes (N=1M, K=16)..."
	python3 ./scripts/gerar_dados.py 1000000 16 $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv

dados_todos: dados_pequenos dados_medios dados_grandes
	@echo ">> Todos os dados de teste gerados"

# rápido com 4 processos
teste: $(PROG) dados_medios
	@echo ">> Teste rápido com 4 processos..."
	@mpirun -np 4 ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6

teste_medio: $(PROG) dados_medios
	@echo ">> dataset médio (100k pontos)"
	@rm -f $(RESULTS_DIR)/resultados_desempenho.csv
	@echo "1 processo:"
	mpirun -np 1 ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_p1.csv $(RESULTS_DIR)/centros_p1.csv $(RESULTS_DIR)/resultados_desempenho.csv
	@echo ""
	@echo "2 processos:"
	mpirun -np 2 ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_p2.csv $(RESULTS_DIR)/centros_p2.csv $(RESULTS_DIR)/resultados_desempenho.csv
	@echo ""
	@echo "4 processos:"
	mpirun -np 4 ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_p4.csv $(RESULTS_DIR)/centros_p4.csv $(RESULTS_DIR)/resultados_desempenho.csv
	@echo ""
	@echo "8 processos:"
	mpirun -np 8 ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_p8.csv $(RESULTS_DIR)/centros_p8.csv $(RESULTS_DIR)/resultados_desempenho.csv


teste_grande: $(PROG) dados_grandes
	@echo ">> dataset grande (1M pontos)"
	@rm -f $(RESULTS_DIR)/resultados_desempenho.csv
	@echo "1 processo:"
	mpirun -np 1 ./$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/assign_p1.csv $(RESULTS_DIR)/centros_p1.csv $(RESULTS_DIR)/resultados_desempenho.csv
	@echo ""
	@echo "2 processos:"
	mpirun -np 2 ./$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/assign_p2.csv $(RESULTS_DIR)/centros_p2.csv $(RESULTS_DIR)/resultados_desempenho.csv
	@echo ""
	@echo "4 processos:"
	mpirun -np 4 ./$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/assign_p4.csv $(RESULTS_DIR)/centros_p4.csv $(RESULTS_DIR)/resultados_desempenho.csv
	@echo ""
	@echo "8 processos:"
	mpirun -np 8 ./$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/assign_p8.csv $(RESULTS_DIR)/centros_p8.csv $(RESULTS_DIR)/resultados_desempenho.csv


teste_tudo: $(PROG) dados_todos
	@echo ">> === TESTE COMPLETO MPI ==="
	@rm -f $(RESULTS_DIR)/resultados_desempenho.csv
	@echo ""
	@echo "[TESTE PEQUENO - 10k pontos]"
	@for P in 1 2 4 8; do \
		echo "  P=$$P processos..."; \
		mpirun -np $$P ./$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/temp_p$$P.csv $(RESULTS_DIR)/temp_c$$P.csv $(RESULTS_DIR)/resultados_desempenho.csv; \
		echo ""; \
	done
	@echo ""
	@echo "[TESTE MÉDIO - 100k pontos]"
	@for P in 1 2 4 8; do \
		echo "  P=$$P processos..."; \
		mpirun -np $$P ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/temp_p$$P.csv $(RESULTS_DIR)/temp_c$$P.csv $(RESULTS_DIR)/resultados_desempenho.csv; \
		echo ""; \
	done
	@echo ""
	@echo "[TESTE GRANDE - 1M pontos]"
	@for P in 1 2 4 8; do \
		echo "  P=$$P processos..."; \
		mpirun -np $$P ./$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/temp_p$$P.csv $(RESULTS_DIR)/temp_c$$P.csv $(RESULTS_DIR)/resultados_desempenho.csv; \
		echo ""; \
	done
	@rm -f $(RESULTS_DIR)/temp_*.csv
	@echo ">> Teste completo!"

# ajustado para máquinas com 4 cores (usa --oversubscribe para P=8)
# teste_tudo: $(PROG) dados_todos
# 	@echo ">> === TESTE COMPLETO MPI ==="
# 	@rm -f $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "[TESTE PEQUENO - 10k pontos]"
# 	@echo "  P=1 processo..."
# 	@mpirun -np 1 ./$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/temp_p1.csv $(RESULTS_DIR)/temp_c1.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "  P=2 processos..."
# 	@mpirun -np 2 ./$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/temp_p2.csv $(RESULTS_DIR)/temp_c2.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "  P=4 processos..."
# 	@mpirun -np 4 ./$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/temp_p4.csv $(RESULTS_DIR)/temp_c4.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "  P=8 processos (oversubscribe)..."
# 	@mpirun --oversubscribe -np 8 ./$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/temp_p8.csv $(RESULTS_DIR)/temp_c8.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "[TESTE MÉDIO - 100k pontos]"
# 	@echo "  P=1 processo..."
# 	@mpirun -np 1 ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/temp_p1.csv $(RESULTS_DIR)/temp_c1.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "  P=2 processos..."
# 	@mpirun -np 2 ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/temp_p2.csv $(RESULTS_DIR)/temp_c2.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "  P=4 processos..."
# 	@mpirun -np 4 ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/temp_p4.csv $(RESULTS_DIR)/temp_c4.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "  P=8 processos (oversubscribe)..."
# 	@mpirun --oversubscribe -np 8 ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/temp_p8.csv $(RESULTS_DIR)/temp_c8.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "[TESTE GRANDE - 1M pontos]"
# 	@echo "  P=1 processo..."
# 	@mpirun -np 1 ./$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/temp_p1.csv $(RESULTS_DIR)/temp_c1.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "  P=2 processos..."
# 	@mpirun -np 2 ./$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/temp_p2.csv $(RESULTS_DIR)/temp_c2.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "  P=4 processos..."
# 	@mpirun -np 4 ./$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/temp_p4.csv $(RESULTS_DIR)/temp_c4.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@echo ""
# 	@echo "  P=8 processos (oversubscribe)..."
# 	@mpirun --oversubscribe -np 8 ./$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/temp_p8.csv $(RESULTS_DIR)/temp_c8.csv $(RESULTS_DIR)/resultados_desempenho.csv
# 	@rm -f $(RESULTS_DIR)/temp_*.csv
# 	@echo ""
# 	@echo ">> Teste completo!"

# Gerar gráficos
graficos: 
	@echo ">> Gerando gráficos a partir dos resultados..."
	python3 ./scripts/grafsmpi.py

# Relatório completo
relatorio: teste_tudo graficos
	@echo ">> Relatório completo gerado"

# Limpar
clean:
	@rm -f $(PROG)
	@rm -f $(RESULTS_DIR)/*.csv
	@echo ">> Arquivos limpos"

cleanall: clean
	@rm -rf $(RESULTS_DIR)
	@echo ">> Tudo limpo"

.PHONY: clean cleanall teste dados_todos graficos relatorio teste_tudo teste_medio teste_grande