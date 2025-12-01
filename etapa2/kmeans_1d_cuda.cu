#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

/* ========== MACROS DE ERRO CUDA ========== */
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error em %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* ========== UTILITÁRIOS CSV (IGUAL AO ORIGINAL) ========== */
static int count_rows(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        exit(1);
    }
    int rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f))
    {
        int only_ws = 1;
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                only_ws = 0;
                break;
            }
        }
        if (!only_ws)
            rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out)
{
    int R = count_rows(path);
    if (R <= 0)
    {
        fprintf(stderr, "Arquivo vazio: %s\n", path);
        exit(1);
    }
    double *A = (double *)malloc((size_t)R * sizeof(double));
    if (!A)
    {
        fprintf(stderr, "Sem memoria para %d linhas\n", R);
        exit(1);
    }
    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        free(A);
        exit(1);
    }
    char line[8192];
    int r = 0;
    while (fgets(line, sizeof(line), f))
    {
        int only_ws = 1;
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                only_ws = 0;
                break;
            }
        }
        if (only_ws)
            continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if (!tok)
        {
            fprintf(stderr, "Linha %d sem valor em %s\n", r + 1, path);
            free(A);
            fclose(f);
            exit(1);
        }
        A[r] = atof(tok);
        r++;
        if (r > R)
            break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N)
{
    if (!path)
        return;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K)
{
    if (!path)
        return;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int c = 0; c < K; c++)
        fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ========== KERNELS CUDA ========== */

/* KERNEL 1: Assignment Step
 * Cada thread processa 1 ponto X[i]
 * Calcula distância para todos K centróides
 * Salva o cluster mais próximo em assign[i]
 * Salva o erro quadrático em sse_per_point[i]
 */
__global__ void kernel_assignment(
    const double *X,       // [N] pontos de entrada
    const double *C,       // [K] centróides
    int *assign,           // [N] saída: cluster de cada ponto
    double *sse_per_point, // [N] erro quadrático de cada ponto
    int N,                 // número de pontos
    int K)                 // número de clusters
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
        return; // thread fora do range

    double xi = X[i];
    int best_cluster = 0;
    double best_dist = 1e300;

    // Varre todos os K centróides
    for (int c = 0; c < K; c++)
    {
        double diff = xi - C[c];
        double dist = diff * diff; // distância euclidiana ao quadrado

        if (dist < best_dist)
        {
            best_dist = dist;
            best_cluster = c;
        }
    }

    assign[i] = best_cluster;
    sse_per_point[i] = best_dist;
}

/* KERNEL 2: Update com Atomics
 * Cada thread adiciona seu ponto à soma do cluster
 * Usa atomicAdd para evitar race conditions
 */
__global__ void kernel_update_atomic(
    const double *X,   // [N] pontos
    const int *assign, // [N] cluster de cada ponto
    double *sum,       // [K] soma dos pontos de cada cluster
    int *cnt,          // [K] contagem de pontos por cluster
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
        return;

    int cluster = assign[i];
    atomicAdd(&sum[cluster], X[i]);
    atomicAdd(&cnt[cluster], 1);
}

/* ========== FUNÇÕES HOST ========== */

/* Calcula SSE total somando os erros individuais */
double compute_sse_host(const double *sse_per_point, int N)
{
    double total = 0.0;
    for (int i = 0; i < N; i++)
    {
        total += sse_per_point[i];
    }
    return total;
}

/* Update na CPU (Opção A - mais simples)
 * Recebe assignments da GPU e calcula médias na CPU
 */
void update_step_host(const double *X, double *C, const int *assign, int N, int K)
{
    double *sum = (double *)calloc((size_t)K, sizeof(double));
    int *cnt = (int *)calloc((size_t)K, sizeof(int));

    for (int i = 0; i < N; i++)
    {
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }

    for (int c = 0; c < K; c++)
    {
        if (cnt[c] > 0)
            C[c] = sum[c] / (double)cnt[c];
        else
            C[c] = X[0]; // cluster vazio recebe primeiro ponto
    }

    free(sum);
    free(cnt);
}

/* Update na GPU (Opção B - com atomics) */
void update_step_device(
    const double *d_X, double *d_C, const int *d_assign,
    int N, int K, int block_size)
{
    // Alocar sum e cnt na GPU
    double *d_sum;
    int *d_cnt;
    CUDA_CHECK(cudaMalloc(&d_sum, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_sum, 0, K * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));

    // Executar kernel de update
    int grid_size = (N + block_size - 1) / block_size;
    kernel_update_atomic<<<grid_size, block_size>>>(d_X, d_assign, d_sum, d_cnt, N);
    CUDA_CHECK(cudaGetLastError());

    // Copiar para CPU e calcular médias
    double *sum = (double *)malloc(K * sizeof(double));
    int *cnt = (int *)malloc(K * sizeof(int));
    CUDA_CHECK(cudaMemcpy(sum, d_sum, K * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cnt, d_cnt, K * sizeof(int), cudaMemcpyDeviceToHost));

    double *C = (double *)malloc(K * sizeof(double));
    for (int c = 0; c < K; c++)
    {
        if (cnt[c] > 0)
            C[c] = sum[c] / (double)cnt[c];
        else
            C[c] = 0.0; // ou outra estratégia
    }

    // Copiar centróides de volta para GPU
    CUDA_CHECK(cudaMemcpy(d_C, C, K * sizeof(double), cudaMemcpyHostToDevice));

    free(sum);
    free(cnt);
    free(C);
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_cnt));
}

/* ========== K-MEANS CUDA ========== */
void kmeans_cuda(
    const double *X, // pontos (host)
    double *C,       // centróides (host)
    int *assign,     // assignments (host)
    int N, int K,
    int max_iter, double eps,
    int block_size,     // threads por bloco
    int use_gpu_update, // 0=CPU update, 1=GPU update
    int *iters_out,
    double *sse_out,
    double *time_h2d,
    double *time_d2h,
    double *time_kernel)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;

    // ========== ALOCAÇÃO GPU ==========
    double *d_X, *d_C, *d_sse_per_point;
    int *d_assign;

    CUDA_CHECK(cudaMalloc(&d_X, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_assign, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sse_per_point, N * sizeof(double)));

    // ========== CÓPIA HOST → DEVICE ==========
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C, K * sizeof(double), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    *time_h2d = ms;

    // ========== LOOP K-MEANS ==========
    int grid_size = (N + block_size - 1) / block_size;
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;

    double total_kernel_time = 0.0;

    for (it = 0; it < max_iter; it++)
    {
        // --- Assignment Step ---
        cudaEventRecord(start);
        kernel_assignment<<<grid_size, block_size>>>(
            d_X, d_C, d_assign, d_sse_per_point, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_kernel_time += ms;

        // --- Calcular SSE ---
        double *sse_host = (double *)malloc(N * sizeof(double));
        CUDA_CHECK(cudaMemcpy(sse_host, d_sse_per_point, N * sizeof(double),
                              cudaMemcpyDeviceToHost));
        sse = compute_sse_host(sse_host, N);
        free(sse_host);

        // --- Critério de parada ---
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps)
        {
            it++;
            break;
        }

        // --- Update Step ---
        if (use_gpu_update)
        {
            // Opção B: update na GPU
            cudaEventRecord(start);
            update_step_device(d_X, d_C, d_assign, N, K, block_size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            total_kernel_time += ms;
        }
        else
        {
            // Opção A: update na CPU
            int *assign_host = (int *)malloc(N * sizeof(int));
            CUDA_CHECK(cudaMemcpy(assign_host, d_assign, N * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            update_step_host(X, C, assign_host, N, K);
            CUDA_CHECK(cudaMemcpy(d_C, C, K * sizeof(double),
                                  cudaMemcpyHostToDevice));
            free(assign_host);
        }

        prev_sse = sse;
    }

    *time_kernel = total_kernel_time;

    // ========== CÓPIA DEVICE → HOST ==========
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C, d_C, K * sizeof(double), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    *time_d2h = ms;

    *iters_out = it;
    *sse_out = sse;

    // ========== LIMPEZA ==========
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_assign));
    CUDA_CHECK(cudaFree(d_sse_per_point));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/* ========== MAIN (COMPATÍVEL COM ETAPA 0) ========== */
int main(int argc, char **argv)
{
    if (argc < 7)
    {
        printf("Uso: %s dados.csv centroides.csv max_iter eps assign.csv centroides.csv [results.csv] [block_size] [gpu_update]\n", argv[0]);
        printf("  Compativel com formato da Etapa 0\n");
        printf("  block_size: threads por bloco (default=256)\n");
        printf("  gpu_update: 0=CPU update, 1=GPU update (default=0)\n");
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = atoi(argv[3]);
    double eps = atof(argv[4]);
    const char *outAssign = argv[5];
    const char *outCentroid = argv[6];
    const char *outResults = (argc > 7) ? argv[7] : NULL;
    int block_size = (argc > 8) ? atoi(argv[8]) : 256;
    int use_gpu_update = (argc > 9) ? atoi(argv[9]) : 0;

    if (max_iter <= 0 || eps <= 0.0)
    {
        fprintf(stderr, "Parametros invalidos: max_iter>0 e eps>0\n");
        return 1;
    }

    // Ler dados
    int N = 0, K = 0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int *)malloc(N * sizeof(int));

    if (!assign)
    {
        fprintf(stderr, "Sem memoria para assign\n");
        free(X);
        free(C);
        return 1;
    }

    // Medir tempo total (incluindo overhead)
    clock_t t0 = clock();

    // Executar k-means
    int iters;
    double sse, time_h2d, time_d2h, time_kernel;

    kmeans_cuda(X, C, assign, N, K, max_iter, eps, block_size, use_gpu_update,
                &iters, &sse, &time_h2d, &time_d2h, &time_kernel);

    clock_t t1 = clock();
    double time_total = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    // Imprimir resultados (formato similar ao original)
    printf("K-means 1D (CUDA)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Block size=%d | Update mode=%s\n", block_size, use_gpu_update ? "GPU" : "CPU");
    printf("Iteracoes: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, time_total);
    printf("  H2D: %.3f ms | Kernels: %.3f ms | D2H: %.3f ms\n",
           time_h2d, time_kernel, time_d2h);

    // Salvar assignments e centróides
    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    // Salvar resultados (formato compatível com Etapa 0)
    if (outResults)
    {
        FILE *f_results = fopen(outResults, "a");
        if (f_results)
        {
            // Verificar se precisa escrever cabeçalho
            fseek(f_results, 0, SEEK_END);
            if (ftell(f_results) == 0)
            {
                fprintf(f_results, "Configuracao,N,K,Iteracoes,SSE,Tempo(ms),BlockSize,UpdateMode,TimeH2D,TimeKernel,TimeD2H\n");
            }
            fprintf(f_results, "CUDA,%d,%d,%d,%.6f,%.1f,%d,%s,%.3f,%.3f,%.3f\n",
                    N, K, iters, sse, time_total, block_size,
                    use_gpu_update ? "GPU" : "CPU",
                    time_h2d, time_kernel, time_d2h);
            fclose(f_results);
            printf("Resultados salvos em %s\n", outResults);
        }
        else
        {
            fprintf(stderr, "Erro ao salvar resultados em %s\n", outResults);
        }
    }

    free(assign);
    free(X);
    free(C);

    return 0;
}