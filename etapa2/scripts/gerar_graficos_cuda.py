import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "resultados"

def gerar_grafs_cuda():
    try:
        # Ler resultados do CSV
        df = pd.read_csv(f'{RESULTS_DIR}/resultados_desempenho.csv')
        
        print("=" * 80)
        print("ANÁLISE DE RESULTADOS - ETAPA 2 (CUDA)")
        print("=" * 80)
        print()
        
        # Separar resultados CUDA e Serial (se existir)
        df_cuda = df[df['Configuracao'] == 'CUDA']
        df_serial = df[df['Configuracao'] != 'CUDA']
        
        if len(df_cuda) == 0:
            print("⚠ Nenhum resultado CUDA encontrado!")
            return
        
        print("Resultados CUDA encontrados:")
        print(df_cuda[['N', 'K', 'BlockSize', 'UpdateMode', 'Iteracoes', 'SSE', 'Tempo(ms)']].to_string(index=False))
        print()
        
        # ============ GRÁFICO 1: Tempos por Dataset ============
        plt.figure(figsize=(12, 6))
        
        # Agrupar por tamanho de dataset
        datasets = []
        labels_dataset = []
        
        for n in sorted(df_cuda['N'].unique()):
            df_n = df_cuda[df_cuda['N'] == n]
            k_val = df_n['K'].iloc[0]
            
            if n < 50000:
                label = f"Pequeno\n({n//1000}k, K={k_val})"
            elif n < 500000:
                label = f"Médio\n({n//1000}k, K={k_val})"
            else:
                label = f"Grande\n({n//1000000}M, K={k_val})"
            
            labels_dataset.append(label)
            datasets.append(df_n['Tempo(ms)'].mean())
        
        bars = plt.bar(labels_dataset, datasets, color=['#4CAF50', '#2196F3', '#FF9800'])
        plt.ylabel('Tempo Médio (ms)', fontsize=12)
        plt.title('Tempos de Execução CUDA por Dataset\n(média entre configurações)', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, tempo in zip(bars, datasets):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(datasets)*0.02, 
                    f'{tempo:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/cuda_tempos_dataset.png', dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {RESULTS_DIR}/cuda_tempos_dataset.png")
        
        # ============ GRÁFICO 2: Comparação Block Sizes ============
        if 'BlockSize' in df_cuda.columns and len(df_cuda['BlockSize'].unique()) > 1:
            plt.figure(figsize=(12, 6))
            
            block_sizes = sorted(df_cuda['BlockSize'].unique())
            n_blocks = len(block_sizes)
            n_datasets = len(df_cuda['N'].unique())
            
            width = 0.25
            x = np.arange(n_datasets)
            
            colors = ['#81C784', '#64B5F6', '#FFB74D']
            
            for i, bs in enumerate(block_sizes):
                df_bs = df_cuda[df_cuda['BlockSize'] == bs]
                tempos = []
                
                for n in sorted(df_cuda['N'].unique()):
                    df_n = df_bs[df_bs['N'] == n]
                    if len(df_n) > 0:
                        tempos.append(df_n['Tempo(ms)'].iloc[0])
                    else:
                        tempos.append(0)
                
                plt.bar(x + i*width, tempos, width, label=f'BS={bs}', color=colors[i % len(colors)])
            
            plt.xlabel('Dataset', fontsize=12)
            plt.ylabel('Tempo (ms)', fontsize=12)
            plt.title('Comparação de Block Sizes - Tempo de Execução', fontsize=14, fontweight='bold')
            plt.xticks(x + width, labels_dataset)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/cuda_block_sizes.png', dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {RESULTS_DIR}/cuda_block_sizes.png")
        
        # ============ GRÁFICO 3: Breakdown de Tempos ============
        if 'TimeH2D' in df_cuda.columns:
            plt.figure(figsize=(12, 6))
            
            # Pegar uma configuração representativa de cada dataset
            labels = []
            time_h2d = []
            time_kernel = []
            time_d2h = []
            
            for n in sorted(df_cuda['N'].unique()):
                df_n = df_cuda[df_cuda['N'] == n]
                # Pegar block_size 256 (mais comum)
                df_rep = df_n[df_n['BlockSize'] == 256]
                if len(df_rep) == 0:
                    df_rep = df_n.iloc[[0]]
                
                k_val = df_rep['K'].iloc[0]
                if n < 50000:
                    label = f"{n//1000}k\nK={k_val}"
                elif n < 500000:
                    label = f"{n//1000}k\nK={k_val}"
                else:
                    label = f"{n//1000000}M\nK={k_val}"
                
                labels.append(label)
                time_h2d.append(df_rep['TimeH2D'].iloc[0])
                time_kernel.append(df_rep['TimeKernel'].iloc[0])
                time_d2h.append(df_rep['TimeD2H'].iloc[0])
            
            x = np.arange(len(labels))
            width = 0.6
            
            p1 = plt.bar(x, time_h2d, width, label='Host→Device', color='#90CAF9')
            p2 = plt.bar(x, time_kernel, width, bottom=time_h2d, label='Kernels', color='#4CAF50')
            p3 = plt.bar(x, time_d2h, width, 
                        bottom=np.array(time_h2d) + np.array(time_kernel), 
                        label='Device→Host', color='#FFB74D')
            
            plt.ylabel('Tempo (ms)', fontsize=12)
            plt.title('Breakdown de Tempos CUDA (Block Size 256)', fontsize=14, fontweight='bold')
            plt.xticks(x, labels)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/cuda_breakdown.png', dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {RESULTS_DIR}/cuda_breakdown.png")
        
        # ============ GRÁFICO 4: Speedup vs Serial ============
        if len(df_serial) > 0:
            plt.figure(figsize=(12, 6))
            
            speedups = []
            labels_speedup = []
            
            for n in sorted(df_cuda['N'].unique()):
                # Tempo serial
                df_ser_n = df_serial[df_serial['N'] == n]
                if len(df_ser_n) == 0:
                    continue
                time_serial = df_ser_n['Tempo(ms)'].iloc[0]
                
                # Tempo CUDA (melhor configuração)
                df_cuda_n = df_cuda[df_cuda['N'] == n]
                time_cuda = df_cuda_n['Tempo(ms)'].min()
                
                speedup = time_serial / time_cuda
                speedups.append(speedup)
                
                k_val = df_cuda_n['K'].iloc[0]
                if n < 50000:
                    label = f"{n//1000}k\nK={k_val}"
                elif n < 500000:
                    label = f"{n//1000}k\nK={k_val}"
                else:
                    label = f"{n//1000000}M\nK={k_val}"
                labels_speedup.append(label)
            
            if len(speedups) > 0:
                bars = plt.bar(labels_speedup, speedups, color=['#66BB6A', '#42A5F5', '#FFA726'])
                plt.ylabel('Speedup (×)', fontsize=12)
                plt.title('Speedup CUDA vs Serial', fontsize=14, fontweight='bold')
                plt.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Baseline (Serial)')
                plt.grid(axis='y', alpha=0.3)
                plt.legend()
                
                for bar, sp in zip(bars, speedups):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups)*0.02, 
                            f'{sp:.1f}×', ha='center', va='bottom', fontweight='bold', fontsize=11)
                
                plt.tight_layout()
                plt.savefig(f'{RESULTS_DIR}/cuda_speedup.png', dpi=300, bbox_inches='tight')
                print(f"✓ Gráfico salvo: {RESULTS_DIR}/cuda_speedup.png")
        
        # ============ GRÁFICO 5: Update Mode Comparison ============
        if 'UpdateMode' in df_cuda.columns and len(df_cuda['UpdateMode'].unique()) > 1:
            plt.figure(figsize=(10, 6))
            
            modes = df_cuda['UpdateMode'].unique()
            n_val = df_cuda['N'].median()  # Dataset médio
            df_mode = df_cuda[df_cuda['N'] == n_val]
            
            if len(df_mode) > 1:
                mode_times = []
                mode_labels = []
                
                for mode in modes:
                    df_m = df_mode[df_mode['UpdateMode'] == mode]
                    if len(df_m) > 0:
                        mode_times.append(df_m['Tempo(ms)'].mean())
                        mode_labels.append(f"{mode} Update")
                
                bars = plt.bar(mode_labels, mode_times, color=['#4CAF50', '#FF9800'])
                plt.ylabel('Tempo Médio (ms)', fontsize=12)
                plt.title(f'Comparação CPU vs GPU Update\n(Dataset médio, N={int(n_val)})', 
                         fontsize=14, fontweight='bold')
                plt.grid(axis='y', alpha=0.3)
                
                for bar, tempo in zip(bars, mode_times):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mode_times)*0.02, 
                            f'{tempo:.1f}ms', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f'{RESULTS_DIR}/cuda_update_modes.png', dpi=300, bbox_inches='tight')
                print(f"✓ Gráfico salvo: {RESULTS_DIR}/cuda_update_modes.png")
        
        # ============ RESUMO TEXTUAL ============
        print()
        print("=" * 80)
        print("RESUMO")
        print("=" * 80)
        
        # Melhor configuração geral
        best_idx = df_cuda['Tempo(ms)'].idxmin()
        best = df_cuda.loc[best_idx]
        print(f"\n🏆 Melhor configuração CUDA:")
        print(f"   N={best['N']}, K={best['K']}")
        print(f"   Block Size: {best['BlockSize']}")
        print(f"   Update Mode: {best['UpdateMode']}")
        print(f"   Tempo: {best['Tempo(ms)']:.2f} ms")
        print(f"   SSE: {best['SSE']:.2f}")
        
        # Média por block size
        if 'BlockSize' in df_cuda.columns:
            print(f"\n📊 Tempo médio por Block Size:")
            for bs in sorted(df_cuda['BlockSize'].unique()):
                avg_time = df_cuda[df_cuda['BlockSize'] == bs]['Tempo(ms)'].mean()
                print(f"   BS={bs}: {avg_time:.2f} ms")
        
        # Speedups
        if len(df_serial) > 0 and len(speedups) > 0:
            print(f"\n⚡ Speedups (CUDA vs Serial):")
            for label, sp in zip(labels_speedup, speedups):
                print(f"   {label.replace(chr(10), ' ')}: {sp:.2f}×")
            print(f"   Speedup médio: {np.mean(speedups):.2f}×")
        
        # Salvar resumo em arquivo
        with open(f'{RESULTS_DIR}/resumo_cuda.txt', 'w') as f:
            f.write("RESUMO DOS RESULTADOS - ETAPA 2 (CUDA)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("MELHOR CONFIGURAÇÃO:\n")
            f.write(f"  N={best['N']}, K={best['K']}\n")
            f.write(f"  Block Size: {best['BlockSize']}\n")
            f.write(f"  Update Mode: {best['UpdateMode']}\n")
            f.write(f"  Tempo: {best['Tempo(ms)']:.2f} ms\n")
            f.write(f"  SSE: {best['SSE']:.2f}\n\n")
            
            f.write("TODOS OS RESULTADOS:\n")
            f.write("-" * 80 + "\n")
            for _, row in df_cuda.iterrows():
                f.write(f"N={row['N']}, K={row['K']}, BS={row['BlockSize']}, Mode={row['UpdateMode']}\n")
                f.write(f"  Tempo: {row['Tempo(ms)']:.2f} ms | SSE: {row['SSE']:.2f} | Iters: {row['Iteracoes']}\n")
                if 'TimeH2D' in row:
                    f.write(f"  H2D: {row['TimeH2D']:.3f}ms | Kernel: {row['TimeKernel']:.3f}ms | D2H: {row['TimeD2H']:.3f}ms\n")
                f.write("\n")
        
        print(f"\n✓ Resumo salvo: {RESULTS_DIR}/resumo_cuda.txt")
        print()
        
    except FileNotFoundError:
        print("❌ Arquivo de resultados não encontrado!")
        print("   Execute primeiro: make teste_todos ou make benchmark")
    except Exception as e:
        print(f"❌ Erro ao gerar gráficos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    gerar_grafs_cuda()