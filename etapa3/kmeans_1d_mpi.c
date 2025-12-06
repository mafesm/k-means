#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* ========== UTILITÁRIOS CSV ========== */
static int count_rows(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        exit(1);
    }
    int rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out)
{
    int R = count_rows(path);
    if (R <= 0) {
        fprintf(stderr, "Arquivo vazio: %s\n", path);
        exit(1);
    }
    double *A = (double *)malloc((size_t)R * sizeof(double));
    if (!A) {
        fprintf(stderr, "Sem memoria para %d linhas\n", R);
        exit(1);
    }
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        free(A);
        exit(1);
    }
    char line[8192];
    int r = 0;
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (only_ws) continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if (!tok) {
            fprintf(stderr, "Linha %d sem valor em %s\n", r + 1, path);
            free(A);
            fclose(f);
            exit(1);
        }
        A[r] = atof(tok);
        r++;
        if (r > R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N)
{
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K)
{
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int c = 0; c < K; c++)
        fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ========== DISTRIBUIÇÃO DE DADOS ========== */
/* Calcula quantos elementos cada processo recebe */
static void calculate_distribution(int N, int P, int rank, int *local_n, int *offset)
{
    int base = N / P;        // Elementos por processo
    int remainder = N % P;   // Elementos restantes
    
    // Processos com rank < remainder recebem 1 elemento extra
    if (rank < remainder) {
        *local_n = base + 1;
        *offset = rank * (base + 1);
    } else {
        *local_n = base;
        *offset = rank * base + remainder;
    }
}

/* ========== ASSIGNMENT LOCAL ========== */
/* Cada processo calcula assignment para seu bloco de pontos */
static double assignment_step_local(
    const double *X_local,    // Pontos locais deste processo
    const double *C,          // Centróides (globais)
    int *assign_local,        // Assignments locais
    int local_n,              // Número de pontos locais
    int K)                    // Número de clusters
{
    double sse_local = 0.0;
    
    for (int i = 0; i < local_n; i++) {
        int best = 0;
        double bestd = 1e300;
        
        for (int c = 0; c < K; c++) {
            double diff = X_local[i] - C[c];
            double d = diff * diff;
            if (d < bestd) {
                bestd = d;
                best = c;
            }
        }
        
        assign_local[i] = best;
        sse_local += bestd;
    }
    
    return sse_local;
}

/* ========== UPDATE LOCAL ========== */
/* Cada processo acumula somas e contagens para seu bloco */
static void update_step_local(
    const double *X_local,
    const int *assign_local,
    double *sum_local,
    int *cnt_local,
    int local_n,
    int K)
{
    // Zerar arrays locais
    for (int c = 0; c < K; c++) {
        sum_local[c] = 0.0;
        cnt_local[c] = 0;
    }
    
    // Acumular localmente
    for (int i = 0; i < local_n; i++) {
        int a = assign_local[i];
        cnt_local[a] += 1;
        sum_local[a] += X_local[i];
    }
}

/* ========== K-MEANS MPI ========== */
void kmeans_mpi(
    const double *X_all,      // Todos os pontos (apenas rank 0 tem inicialmente)
    double *C,                // Centróides (todos os processos)
    int *assign_all,          // Assignments completos (apenas rank 0 preenche)
    int N, int K,
    int max_iter, double eps,
    int rank, int size,       // Rank e tamanho do communicator
    int *iters_out,
    double *sse_out,
    double *time_total,
    double *time_comm)
{
    double t_start, t_end, t_comm_start, t_comm_total = 0.0;
    
    // Calcular distribuição de dados
    int local_n, offset;
    calculate_distribution(N, size, rank, &local_n, &offset);
    
    // Alocar arrays locais
    double *X_local = (double *)malloc(local_n * sizeof(double));
    int *assign_local = (int *)malloc(local_n * sizeof(int));
    double *sum_local = (double *)malloc(K * sizeof(double));
    int *cnt_local = (int *)malloc(K * sizeof(int));
    double *sum_global = (double *)malloc(K * sizeof(double));
    int *cnt_global = (int *)malloc(K * sizeof(int));
    
    if (!X_local || !assign_local || !sum_local || !cnt_local || !sum_global || !cnt_global) {
        fprintf(stderr, "Rank %d: Erro ao alocar memoria\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Arrays para Scatterv/Gatherv
    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        
        for (int p = 0; p < size; p++) {
            int pn, poff;
            calculate_distribution(N, size, p, &pn, &poff);
            sendcounts[p] = pn;
            displs[p] = poff;
        }
    }
    
    // PASSO 0: Distribuir dados iniciais (apenas uma vez)
    t_comm_start = MPI_Wtime();
    MPI_Scatterv(X_all, sendcounts, displs, MPI_DOUBLE,
                 X_local, local_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    t_comm_total += MPI_Wtime() - t_comm_start;
    
    // Início do algoritmo
    t_start = MPI_Wtime();
    
    double prev_sse = 1e300;
    double sse_global = 0.0;
    int it;
    
    for (it = 0; it < max_iter; it++) {
        // PASSO 1: Broadcast dos centróides
        t_comm_start = MPI_Wtime();
        MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        t_comm_total += MPI_Wtime() - t_comm_start;
        
        // PASSO 2: Assignment local
        double sse_local = assignment_step_local(X_local, C, assign_local, local_n, K);
        
        // PASSO 3: Redução do SSE
        t_comm_start = MPI_Wtime();
        MPI_Reduce(&sse_local, &sse_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        // Broadcast do SSE global para todos (necessário para critério de parada)
        MPI_Bcast(&sse_global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        t_comm_total += MPI_Wtime() - t_comm_start;
        
        // PASSO 4: Critério de parada (todos os processos decidem juntos)
        double rel = fabs(sse_global - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps) {
            it++;
            break;
        }
        
        // PASSO 5: Update local (calcular somas e contagens locais)
        update_step_local(X_local, assign_local, sum_local, cnt_local, local_n, K);
        
        // PASSO 6: Allreduce das somas e contagens
        t_comm_start = MPI_Wtime();
        MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        t_comm_total += MPI_Wtime() - t_comm_start;
        
        // PASSO 7: Todos os processos calculam novos centróides
        for (int c = 0; c < K; c++) {
            if (cnt_global[c] > 0)
                C[c] = sum_global[c] / (double)cnt_global[c];
            else
                C[c] = X_local[0]; // Cluster vazio recebe primeiro ponto local
        }
        
        prev_sse = sse_global;
    }
    
    t_end = MPI_Wtime();
    
    // Coletar assignments de volta no rank 0
    t_comm_start = MPI_Wtime();
    MPI_Gatherv(assign_local, local_n, MPI_INT,
                assign_all, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    t_comm_total += MPI_Wtime() - t_comm_start;
    
    *iters_out = it;
    *sse_out = sse_global;
    *time_total = (t_end - t_start) * 1000.0; // converter para ms
    *time_comm = t_comm_total * 1000.0;       // converter para ms
    
    // Liberar memória
    free(X_local);
    free(assign_local);
    free(sum_local);
    free(cnt_local);
    free(sum_global);
    free(cnt_global);
    
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
}

/* ========== MAIN ========== */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 5) {
        if (rank == 0) {
            printf("Uso: mpirun -np P %s dados.csv centroides.csv max_iter eps [assign.csv] [centroides.csv] [results.csv]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = atoi(argv[3]);
    double eps = atof(argv[4]);
    const char *outAssign = (argc > 5) ? argv[5] : "assign_mpi.csv";
    const char *outCentroid = (argc > 6) ? argv[6] : "centroides_mpi.csv";
    const char *outResults = (argc > 7) ? argv[7] : NULL;
    
    if (max_iter <= 0 || eps <= 0.0) {
        if (rank == 0)
            fprintf(stderr, "Parametros invalidos: max_iter>0 e eps>0\n");
        MPI_Finalize();
        return 1;
    }
    
    // Apenas rank 0 lê os dados
    int N = 0, K = 0;
    double *X_all = NULL;
    double *C = NULL;
    int *assign_all = NULL;
    
    if (rank == 0) {
        X_all = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);
        assign_all = (int *)malloc(N * sizeof(int));
        
        if (!assign_all) {
            fprintf(stderr, "Sem memoria para assign\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // Broadcast de N e K para todos os processos
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Processos != 0 alocam memória para centróides
    if (rank != 0) {
        C = (double *)malloc(K * sizeof(double));
        if (!C) {
            fprintf(stderr, "Rank %d: Sem memoria para centroides\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // Executar k-means MPI
    int iters;
    double sse, time_total, time_comm;
    
    kmeans_mpi(X_all, C, assign_all, N, K, max_iter, eps, rank, size,
               &iters, &sse, &time_total, &time_comm);
    
    // Apenas rank 0 imprime resultados e salva arquivos
    if (rank == 0) {
        double time_comp = time_total - time_comm;
        double comm_ratio = (time_comm / time_total) * 100.0;
        
        printf("K-means 1D (MPI)\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
        printf("Processos MPI: %d\n", size);
        printf("Iteracoes: %d | SSE final: %.6f\n", iters, sse);
        printf("Tempo total: %.3f ms\n", time_total);
        printf("  Computacao: %.3f ms (%.1f%%)\n", time_comp, 100.0 - comm_ratio);
        printf("  Comunicacao: %.3f ms (%.1f%%)\n", time_comm, comm_ratio);
        
        // Salvar assignments e centróides
        write_assign_csv(outAssign, assign_all, N);
        write_centroids_csv(outCentroid, C, K);
        
        // Salvar resultados em CSV
        if (outResults) {
            FILE *f = fopen(outResults, "a");
            if (f) {
                fseek(f, 0, SEEK_END);
                if (ftell(f) == 0) {
                    fprintf(f, "Configuracao,N,K,P,Iteracoes,SSE,TempoTotal(ms),TempoComp(ms),TempoComm(ms),CommRatio(%%)\n");
                }
                fprintf(f, "MPI,%d,%d,%d,%d,%.6f,%.3f,%.3f,%.3f,%.2f\n",
                        N, K, size, iters, sse, time_total, time_comp, time_comm, comm_ratio);
                fclose(f);
                printf("Resultados salvos em %s\n", outResults);
            }
        }
        
        free(X_all);
        free(assign_all);
    }
    
    free(C);
    
    MPI_Finalize();
    return 0;
}