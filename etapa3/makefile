# Makefile para K-means 1D MPI - Etapa 3

MPICC = mpicc
CFLAGS = -O2 -std=c99 -lm
PROG = kmeans_1d_mpi
FONTE = kmeans_1d_mpi.c

RESULTS_DIR = resultados

# Compilar
$(PROG): $(FONTE)
	$(MPICC) $(CFLAGS) -o $(PROG) $(FONTE)
	@echo ">> Compilado: $(PROG)"

# Criar diretório de resultados
$(RESULTS_DIR):
	@mkdir -p $(RESULTS_DIR)

# Gerar dados (reutiliza scripts das etapas anteriores)
dados_pequenos: $(RESULTS_DIR)
	@echo ">> Gerando dados pequenos (N=10k, K=4)..."
	python3 ../etapa0/scripts/gerar_dados.py 10000 4 $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv

dados_medios: $(RESULTS_DIR)
	@echo ">> Gerando dados medios (N=100k, K=8)..."
	python3 ../etapa0/scripts/gerar_dados.py 100000 8 $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv

dados_grandes: $(RESULTS_DIR)
	@echo ">> Gerando dados grandes (N=1M, K=16)..."
	python3 ../etapa0/scripts/gerar_dados.py 1000000 16 $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv

dados_todos: dados_pequenos dados_medios dados_grandes
	@echo ">> Todos os dados de teste gerados"

# Teste rápido com 4 processos
teste: $(PROG) dados_medios
	@echo ">> Teste rápido com 4 processos..."
	mpirun -np 4 ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6

# Strong scaling - dataset médio
scaling_medio: $(PROG) dados_medios
	@echo ">> Strong scaling - dataset médio (100k pontos)"
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

# Strong scaling - dataset grande
scaling_grande: $(PROG) dados_grandes
	@echo ">> Strong scaling - dataset grande (1M pontos)"
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

# Benchmark completo: todos os datasets, strong scaling
benchmark: $(PROG) dados_todos
	@echo ">> === BENCHMARK COMPLETO MPI ==="
	@rm -f $(RESULTS_DIR)/resultados_desempenho.csv
	@echo ""
	@echo "[DATASET PEQUENO - 10k pontos]"
	@for P in 1 2 4 8; do \
		echo "  P=$$P processos..."; \
		mpirun -np $$P ./$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/temp_p$$P.csv $(RESULTS_DIR)/temp_c$$P.csv $(RESULTS_DIR)/resultados_desempenho.csv; \
		echo ""; \
	done
	@echo ""
	@echo "[DATASET MÉDIO - 100k pontos]"
	@for P in 1 2 4 8; do \
		echo "  P=$$P processos..."; \
		mpirun -np $$P ./$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/temp_p$$P.csv $(RESULTS_DIR)/temp_c$$P.csv $(RESULTS_DIR)/resultados_desempenho.csv; \
		echo ""; \
	done
	@echo ""
	@echo "[DATASET GRANDE - 1M pontos]"
	@for P in 1 2 4 8; do \
		echo "  P=$$P processos..."; \
		mpirun -np $$P ./$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/temp_p$$P.csv $(RESULTS_DIR)/temp_c$$P.csv $(RESULTS_DIR)/resultados_desempenho.csv; \
		echo ""; \
	done
	@rm -f $(RESULTS_DIR)/temp_*.csv
	@echo ">> Benchmark completo!"

# Gerar gráficos
graficos: 
	@echo ">> Gerando gráficos a partir dos resultados..."
	python3 scripts/gerar_graficos.py

# Relatório completo
relatorio: benchmark graficos
	@echo ">> Relatório completo gerado"

# Limpar
clean:
	@rm -f $(PROG)
	@rm -f $(RESULTS_DIR)/*.csv
	@echo ">> Arquivos limpos"

cleanall: clean
	@rm -rf $(RESULTS_DIR)
	@echo ">> Tudo limpo"

.PHONY: clean cleanall teste dados_todos graficos relatorio benchmark scaling_medio scaling_grande