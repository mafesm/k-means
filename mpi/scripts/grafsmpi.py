import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "resultados"

def gerar_grafs_mpi():
    try:
        # Ler resultados do CSV
        df = pd.read_csv(f'{RESULTS_DIR}/resultados_desempenho.csv')
        
        print("=" * 80)
        print("ANÁLISE DE RESULTADOS - ETAPA 3 (MPI)")
        print("=" * 80)
        print()
        
        # Filtrar apenas resultados MPI
        df_mpi = df[df['Configuracao'] == 'MPI']
        
        if len(df_mpi) == 0:
            print("Nenhum resultado MPI encontrado!")
            return
        
        print("Resultados MPI encontrados:")
        print(df_mpi[['N', 'K', 'P', 'Iteracoes', 'SSE', 'TempoTotal(ms)', 'CommRatio(%)']].to_string(index=False))
        print()
        
        # ============ GRÁFICO 1:  Teste por Dataset ============
        datasets = sorted(df_mpi['N'].unique())
        
        for n in datasets:
            df_n = df_mpi[df_mpi['N'] == n].sort_values('P')
            
            if len(df_n) == 0:
                continue
            
            k_val = df_n['K'].iloc[0]
            
            plt.figure(figsize=(10, 6))
            
            # Tempo total vs número de processos
            plt.plot(df_n['P'], df_n['TempoTotal(ms)'], 
                    'o-', linewidth=2, markersize=8, color='#2196F3', label='Tempo Total')
            
            plt.xlabel('Número de Processos (P)', fontsize=12)
            plt.ylabel('Tempo (ms)', fontsize=12)
            
            if n < 50000:
                title = f'Teste Pequeno\n(N={n//1000}k, K={k_val})'
            elif n < 500000:
                title = f'Teste Médio\n(N={n//1000}k, K={k_val})'
            else:
                title = f'Teste Grande\n(N={n//1000000}M, K={k_val})'
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Adicionar valores nos pontos
            for _, row in df_n.iterrows():
                plt.text(row['P'], row['TempoTotal(ms)'] + max(df_n['TempoTotal(ms)'])*0.03, 
                        f"{row['TempoTotal(ms)']:.1f}ms", 
                        ha='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            
            if n < 50000:
                filename = 'mpi_teste_pequeno.png'
            elif n < 500000:
                filename = 'mpi_teste_medio.png'
            else:
                filename = 'mpi_teste_grande.png'
            
            plt.savefig(f'{RESULTS_DIR}/{filename}', dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {RESULTS_DIR}/{filename}")
        
        # ============ GRÁFICO 2: Speedup ============
        plt.figure(figsize=(12, 6))
        
        colors = ['#4CAF50', '#2196F3', '#FF9800']
        markers = ['o', 's', '^']
        
        for idx, n in enumerate(datasets):
            df_n = df_mpi[df_mpi['N'] == n].sort_values('P')
            
            if len(df_n) == 0:
                continue
            
            # Tempo serial (P=1)
            t_serial = df_n[df_n['P'] == 1]['TempoTotal(ms)'].iloc[0]
            
            # Calcular speedup
            speedups = []
            processes = []
            
            for _, row in df_n.iterrows():
                speedup = t_serial / row['TempoTotal(ms)']
                speedups.append(speedup)
                processes.append(row['P'])
            
            k_val = df_n['K'].iloc[0]
            
            if n < 50000:
                label = f'Pequeno ({n//1000}k, K={k_val})'
            elif n < 500000:
                label = f'Médio ({n//1000}k, K={k_val})'
            else:
                label = f'Grande ({n//1000000}M, K={k_val})'
            
            plt.plot(processes, speedups, 
                    marker=markers[idx % 3], linewidth=2, markersize=8,
                    color=colors[idx % 3], label=label)
        
        # Linha ideal (speedup linear)
        max_p = df_mpi['P'].max()
        plt.plot([1, max_p], [1, max_p], 'k--', linewidth=2, alpha=0.5, label='Ideal (linear)')
        
        plt.xlabel('Número de Processos (P)', fontsize=12)
        plt.ylabel('Speedup (×)', fontsize=12)
        plt.title('Speedup MPI ', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/mpi_speedup.png', dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {RESULTS_DIR}/mpi_speedup.png")
        
        # ============ GRÁFICO 3: Eficiência ============
        plt.figure(figsize=(12, 6))
        
        for idx, n in enumerate(datasets):
            df_n = df_mpi[df_mpi['N'] == n].sort_values('P')
            
            if len(df_n) == 0:
                continue
            
            # Tempo serial (P=1)
            t_serial = df_n[df_n['P'] == 1]['TempoTotal(ms)'].iloc[0]
            
            # Calcular eficiência
            efficiencies = []
            processes = []
            
            for _, row in df_n.iterrows():
                speedup = t_serial / row['TempoTotal(ms)']
                efficiency = (speedup / row['P']) * 100  # em %
                efficiencies.append(efficiency)
                processes.append(row['P'])
            
            k_val = df_n['K'].iloc[0]
            
            if n < 50000:
                label = f'Pequeno ({n//1000}k, K={k_val})'
            elif n < 500000:
                label = f'Médio ({n//1000}k, K={k_val})'
            else:
                label = f'Grande ({n//1000000}M, K={k_val})'
            
            plt.plot(processes, efficiencies, 
                    marker=markers[idx % 3], linewidth=2, markersize=8,
                    color=colors[idx % 3], label=label)
        
        # Linha ideal (100%)
        plt.axhline(y=100, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (100%)')
        
        plt.xlabel('Número de Processos (P)', fontsize=12)
        plt.ylabel('Eficiência (%)', fontsize=12)
        plt.title('Eficiência MPI ', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim([0, 110])
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/mpi_eficiencia.png', dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {RESULTS_DIR}/mpi_eficiencia.png")
        
        # ============ GRÁFICO 4: Breakdown Computação vs Comunicação ============
        # Escolher dataset médio como representativo
        n_medio = sorted(datasets)[len(datasets)//2]
        df_breakdown = df_mpi[df_mpi['N'] == n_medio].sort_values('P')
        
        if len(df_breakdown) > 0:
            plt.figure(figsize=(10, 6))
            
            processes = df_breakdown['P'].values
            time_comp = df_breakdown['TempoComp(ms)'].values
            time_comm = df_breakdown['TempoComm(ms)'].values
            
            x = np.arange(len(processes))
            width = 0.6
            
            p1 = plt.bar(x, time_comp, width, label='Computação', color='#4CAF50')
            p2 = plt.bar(x, time_comm, width, bottom=time_comp, label='Comunicação', color='#FF9800')
            
            plt.ylabel('Tempo (ms)', fontsize=12)
            plt.xlabel('Número de Processos (P)', fontsize=12)
            
            k_val = df_breakdown['K'].iloc[0]
            if n_medio < 50000:
                title = f'Breakdown: Computação vs Comunicação\n(N={n_medio//1000}k, K={k_val})'
            elif n_medio < 500000:
                title = f'Breakdown: Computação vs Comunicação\n(N={n_medio//1000}k, K={k_val})'
            else:
                title = f'Breakdown: Computação vs Comunicação\n(N={n_medio//1000000}M, K={k_val})'
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xticks(x, processes)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Adicionar porcentagem de comunicação
            for i, p in enumerate(processes):
                comm_ratio = df_breakdown[df_breakdown['P'] == p]['CommRatio(%)'].iloc[0]
                total = time_comp[i] + time_comm[i]
                plt.text(i, total + max(time_comp + time_comm)*0.02,
                        f'{comm_ratio:.1f}%\ncomm',
                        ha='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/mpi_breakdown.png', dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {RESULTS_DIR}/mpi_breakdown.png")
        
        # ============ GRÁFICO 5: Overhead de Comunicação ============
        plt.figure(figsize=(12, 6))
        
        for idx, n in enumerate(datasets):
            df_n = df_mpi[df_mpi['N'] == n].sort_values('P')
            
            if len(df_n) == 0:
                continue
            
            processes = df_n['P'].values
            comm_ratios = df_n['CommRatio(%)'].values
            
            k_val = df_n['K'].iloc[0]
            
            if n < 50000:
                label = f'Pequeno ({n//1000}k, K={k_val})'
            elif n < 500000:
                label = f'Médio ({n//1000}k, K={k_val})'
            else:
                label = f'Grande ({n//1000000}M, K={k_val})'
            
            plt.plot(processes, comm_ratios, 
                    marker=markers[idx % 3], linewidth=2, markersize=8,
                    color=colors[idx % 3], label=label)
        
        plt.xlabel('Número de Processos (P)', fontsize=12)
        plt.ylabel('Overhead de Comunicação (%)', fontsize=12)
        plt.title('Overhead de Comunicação MPI', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/mpi_overhead.png', dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {RESULTS_DIR}/mpi_overhead.png")
        
        # ============ RESUMO TEXTUAL ============
        print()
        print("=" * 80)
        print("RESUMO")
        print("=" * 80)
        
        for n in datasets:
            df_n = df_mpi[df_mpi['N'] == n].sort_values('P')
            
            if len(df_n) == 0:
                continue
            
            k_val = df_n['K'].iloc[0]
            
            if n < 50000:
                label = f"Dataset Pequeno (N={n//1000}k, K={k_val})"
            elif n < 500000:
                label = f"Dataset Médio (N={n//1000}k, K={k_val})"
            else:
                label = f"Dataset Grande (N={n//1000000}M, K={k_val})"
            
            print(f"\n {label}:")
            
            t_serial = df_n[df_n['P'] == 1]['TempoTotal(ms)'].iloc[0]
            
            for _, row in df_n.iterrows():
                p = row['P']
                t = row['TempoTotal(ms)']
                speedup = t_serial / t
                efficiency = (speedup / p) * 100
                comm_ratio = row['CommRatio(%)']
                
                print(f"  P={p:2d}: {t:7.2f}ms | Speedup={speedup:5.2f}× | Efic={efficiency:5.1f}% | Comm={comm_ratio:5.1f}%")
        
        # Melhor configuração
        print("\nMelhor Speedup:")
        for n in datasets:
            df_n = df_mpi[df_mpi['N'] == n].sort_values('P')
            if len(df_n) == 0:
                continue
            
            t_serial = df_n[df_n['P'] == 1]['TempoTotal(ms)'].iloc[0]
            speedups = [t_serial / row['TempoTotal(ms)'] for _, row in df_n.iterrows()]
            best_idx = np.argmax(speedups)
            best_row = df_n.iloc[best_idx]
            
            k_val = df_n['K'].iloc[0]
            
            if n < 50000:
                label = f"Pequeno ({n//1000}k, K={k_val})"
            elif n < 500000:
                label = f"Médio ({n//1000}k, K={k_val})"
            else:
                label = f"Grande ({n//1000000}M, K={k_val})"
            
            print(f"  {label}: P={best_row['P']} com speedup de {speedups[best_idx]:.2f}×")
        
        # Salvar resumo em arquivo
        with open(f'{RESULTS_DIR}/resumo_mpi.txt', 'w', encoding='utf-8') as f:
            f.write("RESUMO DOS RESULTADOS - ETAPA 3 (MPI)\n")
            f.write("=" * 80 + "\n\n")
            
            for n in datasets:
                df_n = df_mpi[df_mpi['N'] == n].sort_values('P')
                
                if len(df_n) == 0:
                    continue
                
                k_val = df_n['K'].iloc[0]
                
                if n < 50000:
                    label = f"Dataset Pequeno (N={n//1000}k, K={k_val})"
                elif n < 500000:
                    label = f"Dataset Médio (N={n//1000}k, K={k_val})"
                else:
                    label = f"Dataset Grande (N={n//1000000}M, K={k_val})"
                
                f.write(f"{label}\n")
                f.write("-" * 80 + "\n")
                
                t_serial = df_n[df_n['P'] == 1]['TempoTotal(ms)'].iloc[0]
                
                for _, row in df_n.iterrows():
                    p = row['P']
                    t = row['TempoTotal(ms)']
                    speedup = t_serial / t
                    efficiency = (speedup / p) * 100
                    comm_ratio = row['CommRatio(%)']
                    
                    f.write(f"  P={p:2d}: Tempo={t:7.2f}ms | Speedup={speedup:5.2f}× | Eficiência={efficiency:5.1f}% | Comunicação={comm_ratio:5.1f}%\n")
                
                f.write("\n")
        
        print(f"\n✓ Resumo salvo: {RESULTS_DIR}/resumo_mpi.txt")
        print()
        
    except FileNotFoundError:
        print("Arquivo de resultados não encontrado!")
        print("   Execute primeiro: make benchmark")
    except Exception as e:
        print(f"Erro ao gerar gráficos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    gerar_grafs_mpi()