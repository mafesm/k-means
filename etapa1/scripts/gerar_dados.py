import numpy as np
import sys
import os

def gerar_dados(n_total, k, output_dados, output_centroides):
    np.random.seed(42)  # sempre os mesmo dados para testes reprodutíveis
    
    # Gerar centros dos clusters distribuídos uniformemente
    centros = np.linspace(0, 100 * (k // 4 + 1), k)
    
    # Gerar dados aleatórios em torno dos centros
    dados = []
    pontos_por_cluster = n_total // k
    
    for i, centro in enumerate(centros):
        # Último cluster pega os pontos restantes
        if i == k - 1:
            n_cluster = n_total - len(dados)
        else:
            n_cluster = pontos_por_cluster
        
        # Gerar pontos com desvio padrão proporcional ao número de clusters
        std = 2.0 + (k / 10)  # Aumenta um pouco a variância com K
        cluster_data = np.random.normal(centro, std, n_cluster)
        dados.extend(cluster_data)
    
    # Embaralhar os dados
    dados = np.array(dados)
    np.random.shuffle(dados)
    
    with open(output_dados, 'w') as f:
        f.write("X\n")  # ADICIONAR HEADER
        for valor in dados:
            f.write(f"{valor:.1f}\n")
    
    indices_centroides = np.random.choice(len(dados), k, replace=False)
    centroides_iniciais = dados[indices_centroides]
    
    with open(output_centroides, 'w') as f:
        f.write("centroide\n")  # ADICIONAR HEADER
        for valor in centroides_iniciais:
            f.write(f"{valor:.1f}\n")
    
    print(f"Gerados {n_total} pontos e {k} centroides iniciais")
    print(f"   Arquivos: {output_dados}, {output_centroides}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Uso: python gerar_dados.py <n> <k> <output_dados> <output_centroides>")
        sys.exit(1)
    
    n_total = int(sys.argv[1])
    k = int(sys.argv[2])
    output_dados = sys.argv[3]
    output_centroides = sys.argv[4]
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_dados) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_centroides) or '.', exist_ok=True)
    
    gerar_dados(n_total, k, output_dados, output_centroides)