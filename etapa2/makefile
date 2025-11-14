# Makefile para K-means 1D CUDA - Etapa 2

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60
PROG = kmeans_1d_cuda.exe
FONTE = kmeans_1d_cuda.cu

RESULTS_DIR = resultados

# Configurações de teste
BLOCK_SIZE = 256
GPU_UPDATE = 0

# Criar diretório de resultados
$(RESULTS_DIR):
	@if not exist $(RESULTS_DIR) mkdir $(RESULTS_DIR)

# Compilar
$(PROG): $(FONTE)
	$(NVCC) $(NVCC_FLAGS) -o $(PROG) $(FONTE)
	@echo  >> Compilado: $(PROG)

# Gerar dados (reutiliza scripts da etapa 0)
dados_pequenos: $(RESULTS_DIR)
	@echo  >> gerando dados pequenos (N=10k, K=4)...
	python scripts/gerar_dados.py 10000 4 $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv

dados_medios: $(RESULTS_DIR)
	@echo  >> gerando dados medios (N=100k, K=8)...
	python scripts/gerar_dados.py 100000 8 $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv

dados_grandes: $(RESULTS_DIR)
	@echo  >> gerando dados grandes (N=1M, K=16)...
	python scripts/gerar_dados.py 1000000 16 $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv

dados_todos: dados_pequenos dados_medios dados_grandes
	@echo  >> todos os dados de teste gerados

# Testes individuais com CUDA
teste_pequeno: $(PROG) dados_pequenos
	@echo  >> exe teste pequeno CUDA (N=10k, K=4, BS=$(BLOCK_SIZE))...
	$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/assign_10k_4_cuda.csv $(RESULTS_DIR)/centros_10k_4_cuda.csv $(RESULTS_DIR)/resultados_desempenho.csv $(BLOCK_SIZE) $(GPU_UPDATE)

teste_medio: $(PROG) dados_medios
	@echo  >> exe teste medio CUDA (N=100k, K=8, BS=$(BLOCK_SIZE))...
	$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_100k_8_cuda.csv $(RESULTS_DIR)/centros_100k_8_cuda.csv $(RESULTS_DIR)/resultados_desempenho.csv $(BLOCK_SIZE) $(GPU_UPDATE)

teste_grande: $(PROG) dados_grandes
	@echo  >> exe teste grande CUDA (N=1M, K=16, BS=$(BLOCK_SIZE))...
	$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/assign_1M_16_cuda.csv $(RESULTS_DIR)/centros_1M_16_cuda.csv $(RESULTS_DIR)/resultados_desempenho.csv $(BLOCK_SIZE) $(GPU_UPDATE)

teste_todos: teste_pequeno teste_medio teste_grande
	@echo  >> todos os testes CUDA executados

# Teste com diferentes block sizes
teste_block_sizes: $(PROG) dados_medios
	@echo  >> Testando diferentes block sizes...
	@echo  >> Block size 128:
	$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_bs128.csv $(RESULTS_DIR)/centros_bs128.csv $(RESULTS_DIR)/resultados_desempenho.csv 128 0
	@echo  >> Block size 256:
	$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_bs256.csv $(RESULTS_DIR)/centros_bs256.csv $(RESULTS_DIR)/resultados_desempenho.csv 256 0
	@echo  >> Block size 512:
	$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_bs512.csv $(RESULTS_DIR)/centros_bs512.csv $(RESULTS_DIR)/resultados_desempenho.csv 512 0

# Comparar CPU update vs GPU update
teste_update_modes: $(PROG) dados_medios
	@echo  >> Comparando modos de update...
	@echo  >> CPU Update:
	$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_cpu.csv $(RESULTS_DIR)/centros_cpu.csv $(RESULTS_DIR)/resultados_desempenho.csv 256 0
	@echo  >> GPU Update:
	$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/assign_gpu.csv $(RESULTS_DIR)/centros_gpu.csv $(RESULTS_DIR)/resultados_desempenho.csv 256 1

# Benchmark completo: todos os tamanhos + todas as configurações
benchmark: $(PROG) dados_todos
	@echo  >> === BENCHMARK COMPLETO CUDA ===
	@echo.
	@echo  >> [1/9] Pequeno - BS=128 - CPU update
	$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/temp1.csv $(RESULTS_DIR)/temp1c.csv $(RESULTS_DIR)/resultados_desempenho.csv 128 0
	@echo  >> [2/9] Pequeno - BS=256 - CPU update
	$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/temp2.csv $(RESULTS_DIR)/temp2c.csv $(RESULTS_DIR)/resultados_desempenho.csv 256 0
	@echo  >> [3/9] Pequeno - BS=512 - CPU update
	$(PROG) $(RESULTS_DIR)/dados_10k.csv $(RESULTS_DIR)/centroides_4.csv 100 1e-6 $(RESULTS_DIR)/temp3.csv $(RESULTS_DIR)/temp3c.csv $(RESULTS_DIR)/resultados_desempenho.csv 512 0
	@echo.
	@echo  >> [4/9] Medio - BS=128 - CPU update
	$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/temp4.csv $(RESULTS_DIR)/temp4c.csv $(RESULTS_DIR)/resultados_desempenho.csv 128 0
	@echo  >> [5/9] Medio - BS=256 - CPU update
	$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/temp5.csv $(RESULTS_DIR)/temp5c.csv $(RESULTS_DIR)/resultados_desempenho.csv 256 0
	@echo  >> [6/9] Medio - BS=512 - CPU update
	$(PROG) $(RESULTS_DIR)/dados_100k.csv $(RESULTS_DIR)/centroides_8.csv 100 1e-6 $(RESULTS_DIR)/temp6.csv $(RESULTS_DIR)/temp6c.csv $(RESULTS_DIR)/resultados_desempenho.csv 512 0
	@echo.
	@echo  >> [7/9] Grande - BS=128 - CPU update
	$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/temp7.csv $(RESULTS_DIR)/temp7c.csv $(RESULTS_DIR)/resultados_desempenho.csv 128 0
	@echo  >> [8/9] Grande - BS=256 - CPU update
	$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/temp8.csv $(RESULTS_DIR)/temp8c.csv $(RESULTS_DIR)/resultados_desempenho.csv 256 0
	@echo  >> [9/9] Grande - BS=512 - CPU update
	$(PROG) $(RESULTS_DIR)/dados_1M.csv $(RESULTS_DIR)/centroides_16.csv 100 1e-6 $(RESULTS_DIR)/temp9.csv $(RESULTS_DIR)/temp9c.csv $(RESULTS_DIR)/resultados_desempenho.csv 512 0
	@del /Q $(RESULTS_DIR)\temp*.csv 2>nul
	@echo.
	@echo  >> Benchmark completo!

# Gerar gráficos (adaptado para CUDA)
graficos: 
	@echo  >> gerando graficos a partir dos resultados...
	python scripts/gerar_graficos_cuda.py

# Relatório completo
relatorio: $(PROG) dados_todos benchmark graficos
	@echo  >> relatorio completo gerado

# Verificar instalação CUDA
check_cuda:
	@echo  >> Verificando instalacao CUDA...
	@nvcc --version
	@echo.
	@nvidia-smi

# Limpar
clean:
	@if exist $(PROG) del /Q $(PROG)
	@if exist $(RESULTS_DIR)\*.csv del /Q $(RESULTS_DIR)\*.csv
	@echo  >> Arquivos gerados removidos

cleanall: clean
	@if exist $(RESULTS_DIR) rmdir /S /Q $(RESULTS_DIR)
	@echo  >> Tudo limpo

.PHONY: clean cleanall teste_todos dados_todos graficos relatorio benchmark check_cuda teste_block_sizes teste_update_modes