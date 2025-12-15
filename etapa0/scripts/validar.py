
import csv
import sys
import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class CorrectnessChecker:
    """Verifica corretude dos resultados de K-means com visualização"""
    
    def __init__(self, dados_path, assignments_path, centroids_path, 
                 output_path="corretude.csv", output_dir="resultados"):
        self.dados_path = dados_path
        self.assignments_path = assignments_path
        self.centroids_path = centroids_path
        self.output_path = output_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.dados = None
        self.assignments = None
        self.centroids = None
        self.errors = []
        self.warnings = []
        self.info = {}
    
    def load_data(self):
        """Carrega arquivos de entrada"""
        try:
            # Carregar dados
            self.dados = []
            with open(self.dados_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.dados.append(float(line))
            
            # Carregar assignments
            self.assignments = []
            with open(self.assignments_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.assignments.append(int(line))
            
            # Carregar centróides
            self.centroids = []
            with open(self.centroids_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.centroids.append(float(line))
            
            self.info['N'] = len(self.dados)
            self.info['K'] = len(self.centroids)
            
            return True
        except Exception as e:
            self.errors.append(f"Erro ao carregar dados: {e}")
            return False
    
    def check_assignment_bounds(self):
        """Verifica se assignments estão dentro dos limites [0, K-1]"""
        K = len(self.centroids)
        for i, assign in enumerate(self.assignments):
            if assign < 0 or assign >= K:
                self.errors.append(f"Assignment[{i}] = {assign} inválido (K={K})")
    
    def check_assignment_count(self):
        """Verifica se cada ponto tem um assignment"""
        if len(self.assignments) != len(self.dados):
            self.errors.append(
                f"Número de assignments ({len(self.assignments)}) "
                f"!= número de dados ({len(self.dados)})"
            )
    
    def check_centroid_validity(self):
        """Verifica se centróides estão em posições válidas"""
        for i, centroid in enumerate(self.centroids):
            min_val = min(self.dados)
            max_val = max(self.dados)
            
            if centroid < min_val * 1.5 or centroid > max_val * 1.5:
                self.warnings.append(
                    f"Centroide[{i}] = {centroid:.2f} fora do range "
                    f"[{min_val:.2f}, {max_val:.2f}]"
                )
    
    def check_non_empty_clusters(self):
        """Verifica se há clusters vazios"""
        K = len(self.centroids)
        cluster_counts = [0] * K
        
        for assign in self.assignments:
            cluster_counts[assign] += 1
        
        self.info['clusters_vazios'] = sum(1 for c in cluster_counts if c == 0)
        self.info['cluster_min_size'] = min(cluster_counts) if cluster_counts else 0
        self.info['cluster_max_size'] = max(cluster_counts) if cluster_counts else 0
        
        for i, count in enumerate(cluster_counts):
            if count == 0:
                self.warnings.append(f"Cluster {i} está vazio")
    
    def compute_sse(self):
        """Calcula SSE a partir dos dados e assignments"""
        sse = 0.0
        for i, data_point in enumerate(self.dados):
            cluster_id = self.assignments[i]
            centroid = self.centroids[cluster_id]
            error = data_point - centroid
            sse += error * error
        
        self.info['SSE_computed'] = sse
        return sse
    
    def check_centroid_convergence(self):
        """Verifica se cada centróide é realmente a média do seu cluster"""
        K = len(self.centroids)
        
        for k in range(K):
            cluster_points = [
                self.dados[i] for i in range(len(self.dados))
                if self.assignments[i] == k
            ]
            
            if not cluster_points:
                continue
            
            real_mean = sum(cluster_points) / len(cluster_points)
            reported_centroid = self.centroids[k]
            
            tolerance = 0.01 * abs(real_mean) + 1e-6
            
            if abs(real_mean - reported_centroid) > tolerance:
                self.warnings.append(
                    f"Centroide[{k}]: esperado {real_mean:.6f}, "
                    f"obtido {reported_centroid:.6f}"
                )
    
    def check_cluster_assignment_optimality(self):
        """Verifica se cada ponto está atribuído ao cluster mais próximo"""
        errors_count = 0
        
        for i, data_point in enumerate(self.dados):
            assigned_cluster = self.assignments[i]
            assigned_centroid = self.centroids[assigned_cluster]
            
            dist_to_assigned = abs(data_point - assigned_centroid)
            
            for k in range(len(self.centroids)):
                dist_to_k = abs(data_point - self.centroids[k])
                if dist_to_k < dist_to_assigned - 1e-6:
                    errors_count += 1
                    if errors_count <= 3:
                        self.errors.append(
                            f"Ponto[{i}]={data_point:.2f}: atribuído a cluster {assigned_cluster} "
                            f"(dist={dist_to_assigned:.2f}), mas cluster {k} é mais próximo "
                            f"(dist={dist_to_k:.2f})"
                        )
        
        if errors_count > 3:
            self.errors.append(f"... e mais {errors_count - 3} erros de atribuição")
        
        self.info['assignment_errors'] = errors_count
    
    # ========== VISUALIZAÇÕES ==========
    
    def plot_clustering_visualization(self):
        """Gera gráficos visuais dos clusters"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Visualização de Corretude - K-means 1D', fontsize=16, fontweight='bold')
        
        K = len(self.centroids)
        colors = plt.cm.tab20(np.linspace(0, 1, max(K, 3)))
        
        # ===== Gráfico 1: Dados sem clustering (topo esquerdo) =====
        ax1 = axes[0, 0]
        ax1.scatter(range(len(self.dados)), self.dados, alpha=0.6, s=30, color='gray', label='Dados')
        ax1.set_xlabel('Índice do Ponto', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Valor', fontsize=11, fontweight='bold')
        ax1.set_title('Dados Originais (antes do K-means)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # ===== Gráfico 2: Clusters com cores (topo direito) =====
        ax2 = axes[0, 1]
        for k in range(K):
            mask = np.array(self.assignments) == k
            indices = np.where(mask)[0]
            values = np.array(self.dados)[mask]
            ax2.scatter(indices, values, alpha=0.7, s=50, color=colors[k], label=f'Cluster {k}')
        
        # Marcar centróides
        for k, centroid in enumerate(self.centroids):
            ax2.axhline(y=centroid, color=colors[k], linestyle='--', linewidth=2.5, alpha=0.8)
            ax2.text(0, centroid, f'  C{k}={centroid:.2f}', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor=colors[k], alpha=0.3))
        
        ax2.set_xlabel('Índice do Ponto', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Valor', fontsize=11, fontweight='bold')
        ax2.set_title('Resultado do K-means (com clusters)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        
        # ===== Gráfico 3: Distribuição por cluster (inferior esquerdo) =====
        ax3 = axes[1, 0]
        cluster_sizes = [sum(1 for a in self.assignments if a == k) for k in range(K)]
        bars = ax3.bar(range(K), cluster_sizes, color=colors[:K], alpha=0.7, edgecolor='black', linewidth=2)
        
        # Adicionar valores nas barras
        for bar, size in zip(bars, cluster_sizes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(size)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax3.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Tamanho do Cluster', fontsize=11, fontweight='bold')
        ax3.set_title('Distribuição de Tamanhos (Cluster Balance)', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_xticks(range(K))
        
        # ===== Gráfico 4: Distâncias para centróide (inferior direito) =====
        ax4 = axes[1, 1]
        
        distances = []
        for i, data_point in enumerate(self.dados):
            centroid = self.centroids[self.assignments[i]]
            dist = abs(data_point - centroid)
            distances.append(dist)
        
        ax4.hist(distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2.5, label=f'Média: {np.mean(distances):.2f}')
        ax4.axvline(np.median(distances), color='green', linestyle='--', linewidth=2.5, label=f'Mediana: {np.median(distances):.2f}')
        
        ax4.set_xlabel('Distância ao Centróide', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequência', fontsize=11, fontweight='bold')
        ax4.set_title('Distribuição de Distâncias (SSE/N)', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'visualizacao_clusters.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nGráfico de clusters salvo: {plot_path}")
        plt.close()
    
    def plot_convergence_check(self):
        """Gera gráfico verificando convergência do centróide"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle('Verificação de Convergência - Centróides', fontsize=16, fontweight='bold')
        
        K = len(self.centroids)
        colors = plt.cm.tab20(np.linspace(0, 1, max(K, 3)))
        
        # ===== Gráfico 1: Centróide vs Média Real =====
        ax1 = axes[0]
        real_means = []
        x_pos = np.arange(K)
        
        for k in range(K):
            cluster_points = [
                self.dados[i] for i in range(len(self.dados))
                if self.assignments[i] == k
            ]
            if cluster_points:
                real_mean = sum(cluster_points) / len(cluster_points)
            else:
                real_mean = 0
            real_means.append(real_mean)
        
        width = 0.35
        bars1 = ax1.bar(x_pos - width/2, self.centroids, width, label='Centróide Reportado', 
                       color='steelblue', alpha=0.7, edgecolor='black')
        bars2 = ax1.bar(x_pos + width/2, real_means, width, label='Média Real do Cluster', 
                       color='coral', alpha=0.7, edgecolor='black')
        
        ax1.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Valor', fontsize=11, fontweight='bold')
        ax1.set_title('Centróides: Reportado vs Real\n(Devem ser iguais)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # ===== Gráfico 2: Erro de Convergência =====
        ax2 = axes[1]
        errors = []
        for k in range(K):
            cluster_points = [
                self.dados[i] for i in range(len(self.dados))
                if self.assignments[i] == k
            ]
            if cluster_points:
                real_mean = sum(cluster_points) / len(cluster_points)
                error = abs(self.centroids[k] - real_mean)
            else:
                error = 0
            errors.append(error)
        
        bars = ax2.bar(range(K), errors, color=colors[:K], alpha=0.7, edgecolor='black', linewidth=2)
        
        # Adicionar valores nas barras
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{err:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Erro Absoluto', fontsize=11, fontweight='bold')
        ax2.set_title('Erro de Convergência por Cluster\n(Próximo a 0 = Convergido)', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticks(range(K))
        
        plt.tight_layout()
        plot_path = self.output_dir / 'verificacao_convergencia.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de convergência salvo: {plot_path}")
        plt.close()
    
    def plot_sse_analysis(self):
        """Gera gráfico de análise SSE"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle('Análise de SSE (Sum of Squared Errors)', fontsize=16, fontweight='bold')
        
        K = len(self.centroids)
        colors = plt.cm.tab20(np.linspace(0, 1, max(K, 3)))
        
        # ===== Gráfico 1: SSE por cluster =====
        ax1 = axes[0]
        sse_per_cluster = []
        
        for k in range(K):
            sse_k = 0.0
            for i, data_point in enumerate(self.dados):
                if self.assignments[i] == k:
                    centroid = self.centroids[k]
                    error = data_point - centroid
                    sse_k += error * error
            sse_per_cluster.append(sse_k)
        
        bars = ax1.bar(range(K), sse_per_cluster, color=colors[:K], alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, sse in zip(bars, sse_per_cluster):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{sse:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax1.set_ylabel('SSE do Cluster', fontsize=11, fontweight='bold')
        ax1.set_title('Contribuição ao SSE Total por Cluster', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticks(range(K))
        
        # ===== Gráfico 2: Distribuição de erros =====
        ax2 = axes[1]
        squared_errors = []
        for i, data_point in enumerate(self.dados):
            centroid = self.centroids[self.assignments[i]]
            error = data_point - centroid
            squared_errors.append(error ** 2)
        
        ax2.hist(squared_errors, bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(squared_errors), color='red', linestyle='--', linewidth=2.5, 
                   label=f'Média: {np.mean(squared_errors):.2f}')
        
        total_sse = sum(squared_errors)
        ax2.text(0.95, 0.95, f'SSE Total: {total_sse:.2f}\nSSE/N: {total_sse/len(self.dados):.2f}',
                transform=ax2.transAxes, fontsize=12, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_xlabel('Erro Quadrático (error²)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequência', fontsize=11, fontweight='bold')
        ax2.set_title('Distribuição de Erros Quadráticos', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'analise_sse.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de SSE salvo: {plot_path}")
        plt.close()
    

    
    def run_all_checks(self):
        """Executa todas as verificações"""
        if not self.load_data():
            return False
        
        print(f"\n{'='*70}")
        print(f"VERIFICAÇÃO DE CORRETUDE - K-means 1D (SERIAL)")
        print(f"{'='*70}\n")
        
        print(f"Dados carregados:")
        print(f"  N = {self.info['N']:,} pontos")
        print(f"  K = {self.info['K']} clusters")
        print(f"  Min: {min(self.dados):.2f}, Max: {max(self.dados):.2f}")
        print(f"  Range: {max(self.dados) - min(self.dados):.2f}\n")
        
        self.check_assignment_count()
        self.check_assignment_bounds()
        self.check_centroid_validity()
        self.check_non_empty_clusters()
        
        sse = self.compute_sse()
        print(f"SSE Calculado: {sse:.6f}")
        
        self.check_centroid_convergence()
        self.check_cluster_assignment_optimality()
        
        # Resumo
        print(f"\n{'─'*70}")
        print(f"RESUMO DA VERIFICAÇÃO")
        print(f"{'─'*70}")
        
        if self.info.get('clusters_vazios', 0) > 0:
            print(f" {self.info['clusters_vazios']} clusters vazios")
        
        print(f"✓ Distribuição de clusters:")
        print(f"    Min tamanho: {self.info['cluster_min_size']}")
        print(f"    Max tamanho: {self.info['cluster_max_size']}")
        
        if self.errors:
            print(f"\nERROS ENCONTRADOS ({len(self.errors)}):")
            for err in self.errors[:5]:
                print(f"   • {err}")
            if len(self.errors) > 5:
                print(f"   ... e mais {len(self.errors) - 5}")
        
        if self.warnings:
            print(f"\n AVISOS ({len(self.warnings)}):")
            for warn in self.warnings[:5]:
                print(f"   • {warn}")
            if len(self.warnings) > 5:
                print(f"   ... e mais {len(self.warnings) - 5}")
        
        print(f"\n{'='*70}")
        status = "CORRETO" if not self.errors else "INCORRETO"
        print(f"{status}")
        print(f"{'='*70}\n")
        
        return len(self.errors) == 0
    
    def save_results(self):
        """Salva resultados em CSV"""
        try:
            results = {
                'N': self.info.get('N', 0),
                'K': self.info.get('K', 0),
                'SSE_computed': self.info.get('SSE_computed', 0),
                'clusters_vazios': self.info.get('clusters_vazios', 0),
                'cluster_min_size': self.info.get('cluster_min_size', 0),
                'cluster_max_size': self.info.get('cluster_max_size', 0),
                'assignment_errors': self.info.get('assignment_errors', 0),
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'status': 'CORRETO' if not self.errors else 'INCORRETO'
            }
            
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results.keys())
                writer.writeheader()
                writer.writerow(results)
            
            print(f"Resultados salvos em: {self.output_path}")
        except Exception as e:
            print(f"Erro ao salvar resultados: {e}")
    
    def generate_all_plots(self):
        """Gera todos os gráficos"""
        print(f"\n{'='*70}")
        print(f"GERANDO GRÁFICOS DE VISUALIZAÇÃO")
        print(f"{'='*70}\n")
        
        print("Gerando gráficos...")
        self.plot_clustering_visualization()
        self.plot_convergence_check()
        self.plot_sse_analysis()
        
        print(f"\n{'='*70}")
        print(f"TODOS OS GRÁFICOS GERADOS")
        print(f"{'='*70}")
        print(f"\nGráficos salvos em: {self.output_dir}/")
        print(f"   ✓ visualizacao_clusters.png")
        print(f"   ✓ verificacao_convergencia.png")
        print(f"   ✓ analise_sse.png")
        print(f"   ✓ resumo_corretude.png\n")

def main():
    if len(sys.argv) < 4:
        print(f"Uso: {sys.argv[0]} <dados.csv> <assignments.csv> <centroides.csv> [output.csv]")
        sys.exit(1)
    
    dados_path = sys.argv[1]
    assignments_path = sys.argv[2]
    centroids_path = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 else "corretude.csv"
    
    checker = CorrectnessChecker(dados_path, assignments_path, centroids_path, output_path)
    success = checker.run_all_checks()
    checker.save_results()
    checker.generate_all_plots()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
