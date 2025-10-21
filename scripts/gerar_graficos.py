import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "resultados"

def gerar_grafs():
    try:
        # Ler resultados do CSV
        df = pd.read_csv(f'{RESULTS_DIR}/resultados_desempenho.csv')
        
        # Criar coluna de configuração para legenda
        df['Configuracao'] = df.apply(
            lambda row: f"Pequeno\n({row['N']//1000}k, {row['K']})" if row['N'] == 10000 else 
                       f"Medio\n({row['N']//1000}k, {row['K']})" if row['N'] == 100000 else 
                       f"Grande\n({row['N']//1000//1000}M, {row['K']})", 
            axis=1
        )
        
        print(df[['Configuracao', 'N', 'K', 'Iteracoes', 'SSE', 'Tempo(ms)']])
        
        # Gráfico de tempos
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['Configuracao'], df['Tempo(ms)'], 
                      color=['#4CAF50', '#2196F3', '#FF9800'])
        
        plt.ylabel('Tempo de Execucao (ms)', fontsize=12)
        plt.title('Tempos de Execucao - K-means 1D OpenMP\n(Etapa 1 - CPU)', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, tempo in zip(bars, df['Tempo(ms)']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['Tempo(ms)'])*0.01, 
                    f'{tempo:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/tempos_execucao.png', dpi=300, bbox_inches='tight')
        
        # Gráfico de SSE
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['Configuracao'], df['SSE'], 
                      color=['#81C784', '#64B5F6', '#FFB74D'])
        
        plt.ylabel('SSE Final', fontsize=12)
        plt.title('SSE Final por Configuracao - K-means 1D OpenMP', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, sse in zip(bars, df['SSE']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['SSE'])*0.01, 
                    f'{sse:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/sse_final.png', dpi=300, bbox_inches='tight')
        
        # Gráfico de iterações
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['Configuracao'], df['Iteracoes'], 
                      color=['#A5D6A7', '#90CAF9', '#FFCC80'])
        
        plt.ylabel('Numero de Iteracoes', fontsize=12)
        plt.title('Iteracoes ate Convergencia - K-means 1D OpenMP', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, iters in zip(bars, df['Iteracoes']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['Iteracoes'])*0.01, 
                    f'{iters}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/iteracoes.png', dpi=300, bbox_inches='tight')
        
        print("\nGraficos gerados:")
        print(f"   {RESULTS_DIR}/tempos_execucao.png")
        print(f"   {RESULTS_DIR}/sse_final.png")
        print(f"   {RESULTS_DIR}/iteracoes.png")
        
        # Salvar tabela resumo
        with open(f'{RESULTS_DIR}/resumo_resultados.txt', 'w') as f:
            f.write("RESUMO DOS RESULTADOS - ETAPA 0\n")
            f.write("===============================\n\n")
            for _, row in df.iterrows():
                f.write(f"{row['Configuracao']}:\n")
                f.write(f"  Tempo: {row['Tempo(ms)']:.1f} ms\n")
                f.write(f"  SSE: {row['SSE']:.2f}\n")
                f.write(f"  Iteracoes: {row['Iteracoes']}\n")
                f.write(f"  N={row['N']}, K={row['K']}\n\n")
        
        print(f"   {RESULTS_DIR}/resumo_resultados.txt")
        
    except FileNotFoundError:
        print("Arquivo de resultados nao encontrado.")
        print("   Execute primeiro: make teste_todos")
    except Exception as e:
        print(f"Erro ao gerar graficos: {e}")

if __name__ == "__main__":
    gerar_grafs()