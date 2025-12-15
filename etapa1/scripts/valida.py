#!/usr/bin/env python3
"""
Script de validação de corretude para K-means 1D com OpenMP
Gera gráficos de visualização dos clusters e análises detalhadas
"""

import sys
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def main():
    if len(sys.argv) < 5:
        print("Uso: python valida.py <dados.csv> <assign.csv> <centros.csv> <saida.csv>")
        sys.exit(1)

    arquivo_dados = sys.argv[1]
    arquivo_assign = sys.argv[2]
    arquivo_centroides = sys.argv[3]
    arquivo_saida = sys.argv[4]

    print("\n" + "="*70)
    print("VALIDAÇÃO DE CORRETUDE - K-means 1D (COM VISUALIZAÇÕES)")
    print("="*70 + "\n")

    # ========== CARREGAMENTO DE DADOS ==========
    try:
        print(f"[CARREGANDO] {arquivo_dados}")
        dados = []
        with open(arquivo_dados, 'r') as f:
            leitor = csv.reader(f)
            next(leitor)  # Pula header
            dados = [float(row[0]) for row in leitor]
        N = len(dados)
        print(f" ✓ N = {N:,} pontos\n")
    except Exception as e:
        print(f" ✗ ERRO: {e}\n")
        sys.exit(1)

    try:
        print(f"[CARREGANDO] {arquivo_assign}")
        assignments = []
        with open(arquivo_assign, 'r') as f:
            leitor = csv.reader(f)
            next(leitor)  # Pula header
            assignments = [int(row[0]) for row in leitor]
        
        if len(assignments) != N:
            print(f" ✗ ERRO: Assignments ({len(assignments)}) != Dados ({N})\n")
            sys.exit(1)
        
        K = max(assignments) + 1
        print(f" ✓ {N:,} assignments carregados")
        print(f" ✓ K = {K} clusters\n")
    except Exception as e:
        print(f" ✗ ERRO: {e}\n")
        sys.exit(1)

    try:
        print(f"[CARREGANDO] {arquivo_centroides}")
        centroides = []
        with open(arquivo_centroides, 'r') as f:
            leitor = csv.reader(f)
            next(leitor)  # Pula header
            centroides = [float(row[0]) for row in leitor]
        print(f" ✓ {len(centroides)} centróides carregados\n")
    except Exception as e:
        print(f" ✗ ERRO: {e}\n")
        sys.exit(1)

    # ========== VALIDAÇÕES ==========
    print("[VALIDAÇÕES]")
    erros = 0

    # 1. Valida assignments
    for i, assign in enumerate(assignments):
        if assign < 0 or assign >= K:
            erros += 1

    if erros > 0:
        print(f" ✗ {erros} assignments inválidos")
    else:
        print(f" ✓ Todos os assignments em [0, {K-1}]")

    # 2. Calcula SSE
    sse = 0.0
    for i in range(N):
        k = assignments[i]
        if k < len(centroides):
            centroide = centroides[k]
            erro = dados[i] - centroide
            sse += erro * erro
        else:
            print(f" ✗ Assignment {k} fora dos centróides ({len(centroides)})")
            erros += 1

    print(f" ✓ SSE = {sse:.6f}")
    print(f" ✓ SSE/N = {sse/N:.6f}")

    # 3. Analisa distribuição
    tamanhos = [0] * K
    for assign in assignments:
        if assign < K:
            tamanhos[assign] += 1

    tamanho_min = min(tamanhos) if tamanhos else 0
    tamanho_max = max(tamanhos) if tamanhos else 0
    razao = tamanho_max / tamanho_min if tamanho_min > 0 else 0
    clusters_vazios = sum(1 for t in tamanhos if t == 0)

    print(f" ✓ Tamanhos: min={tamanho_min}, max={tamanho_max}")
    print(f" ✓ Razão: {razao:.2f}x")

    if clusters_vazios > 0:
        print(f" ✗ {clusters_vazios} clusters vazios")
        erros += clusters_vazios

    # Status
    status = "CORRETO" if erros == 0 else "ERRO"

    # ========== VISUALIZAÇÕES ==========
    print(f"\n[GERANDO GRÁFICOS]")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Validação de Corretude - K-means 1D (N={N:,}, K={K})', 
                     fontsize=16, fontweight='bold')

        colors = plt.cm.tab20(np.linspace(0, 1, max(K, 3)))

        # ===== Gráfico 1: Dados originais =====
        ax1 = axes[0, 0]
        ax1.scatter(range(len(dados)), dados, alpha=0.6, s=20, color='gray', label='Dados')
        ax1.set_xlabel('Índice', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Valor', fontsize=11, fontweight='bold')
        ax1.set_title('Dados Originais', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # ===== Gráfico 2: Clusters com cores =====
        ax2 = axes[0, 1]
        for k in range(K):
            mask = np.array(assignments) == k
            indices = np.where(mask)[0]
            values = np.array(dados)[mask]
            ax2.scatter(indices, values, alpha=0.7, s=40, color=colors[k], label=f'C{k}')

            # Marcar centróide
            centroide = centroides[k]
            ax2.axhline(y=centroide, color=colors[k], linestyle='--', linewidth=2, alpha=0.8)

        ax2.set_xlabel('Índice', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Valor', fontsize=11, fontweight='bold')
        ax2.set_title('Resultado K-means (com centróides)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', ncol=4, fontsize=9)

        # ===== Gráfico 3: Distribuição por cluster =====
        ax3 = axes[1, 0]
        bars = ax3.bar(range(K), tamanhos, color=colors[:K], alpha=0.7, 
                       edgecolor='black', linewidth=2)

        # Adicionar valores nas barras
        for bar, size in zip(bars, tamanhos):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(size)}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)

        ax3.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Tamanho', fontsize=11, fontweight='bold')
        ax3.set_title('Distribuição de Tamanhos (Cluster Balance)', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_xticks(range(K))
        ax3.axhline(y=N/K, color='red', linestyle=':', linewidth=2, 
                   label=f'Ideal: {N/K:.0f}')
        ax3.legend()

        # ===== Gráfico 4: Métricas de qualidade =====
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Texto com métricas
        metrics_text = f"""
MÉTRICAS DE CORRETUDE

Número de Dados (N):        {N:,}
Número de Clusters (K):     {K}

SSE Total:                  {sse:,.2f}
SSE por Ponto:              {sse/N:,.2f}

Cluster Mínimo:             {tamanho_min} pontos
Cluster Máximo:             {tamanho_max} pontos
Razão Máx/Mín:              {razao:.2f}x

Clusters Vazios:            {clusters_vazios}
Erros de Atribuição:        {erros}

STATUS:                     {'✓ CORRETO' if status == 'CORRETO' else '✗ ERRO'}
        """

        # Cores baseadas no status
        box_color = 'lightgreen' if status == 'CORRETO' else 'lightcoral'

        ax4.text(0.1, 0.95, metrics_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

        plt.tight_layout()
        
        # Salvar figura
        output_dir = Path("resultados")
        output_dir.mkdir(exist_ok=True)
        
        # Extrair base do nome do arquivo
        base_name = Path(arquivo_dados).stem
        fig_path = output_dir / f"validacao_{base_name}.png"
        
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f" ✓ Gráfico salvo em {fig_path}")
        
        plt.close()

    except Exception as e:
        print(f" ✗ Erro ao gerar gráficos: {e}")

    # ========== SALVAR RELATÓRIO ==========
    print(f"\n[SAÍDA] {arquivo_saida}")
    try:
        with open(arquivo_saida, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'N', 'K', 'SSE_computed', 'clusters_vazios',
                'cluster_min_size', 'cluster_max_size',
                'assignment_errors', 'total_errors', 'total_warnings', 'status'
            ])
            writer.writerow([
                N, K, f"{sse:.6f}", clusters_vazios,
                tamanho_min, tamanho_max, erros, erros, 0, status
            ])
        print(f" ✓ Relatório salvo\n")
    except Exception as e:
        print(f" ✗ ERRO ao salvar: {e}\n")
        sys.exit(1)

    # ========== RESULTADO FINAL ==========
    print("="*70)
    print(f"STATUS: {'✓ CORRETO' if status == 'CORRETO' else '✗ ERRO'}")
    print("="*70 + "\n")

    sys.exit(0 if status == "CORRETO" else 1)


if __name__ == '__main__':
    main()