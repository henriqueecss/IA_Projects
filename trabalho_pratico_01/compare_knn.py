# compare_knn.py
import time
import matplotlib.pyplot as plt
import numpy as np
from knn_hardcore import evaluate_multiple_k_values
from knn_sklearn import evaluate_multiple_k_values_sklearn

def plot_metrics(k_values, hc_results, sklearn_results):
    # Extrai métricas
    hc_acc = [r['accuracy'] for r in hc_results]
    hc_prec = [r['precision'] for r in hc_results]
    hc_rec = [r['recall'] for r in hc_results]

    sk_acc = [r['accuracy'] for r in sklearn_results]
    sk_prec = [r['precision'] for r in sklearn_results]
    sk_rec = [r['recall'] for r in sklearn_results]

    x = np.arange(len(k_values))  # posição das barras
    width = 0.35  # largura das barras

    # Acurácia
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, hc_acc, width, label='Hardcore')
    plt.bar(x + width/2, sk_acc, width, label='Sklearn')
    plt.xticks(x, k_values)
    plt.ylim(0, 1)
    plt.xlabel('k')
    plt.ylabel('Acurácia')
    plt.title('Comparação de Acurácia')
    plt.legend()
    plt.show()

    # Precisão
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, hc_prec, width, label='Hardcore')
    plt.bar(x + width/2, sk_prec, width, label='Sklearn')
    plt.xticks(x, k_values)
    plt.ylim(0, 1)
    plt.xlabel('k')
    plt.ylabel('Precisão')
    plt.title('Comparação de Precisão')
    plt.legend()
    plt.show()

    # Revocação
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, hc_rec, width, label='Hardcore')
    plt.bar(x + width/2, sk_rec, width, label='Sklearn')
    plt.xticks(x, k_values)
    plt.ylim(0, 1)
    plt.xlabel('k')
    plt.ylabel('Revocação')
    plt.title('Comparação de Revocação')
    plt.legend()
    plt.show()

def plot_times(hc_time, sklearn_time):
    classifiers = ['Hardcore', 'Sklearn']
    times = [hc_time, sklearn_time]

    plt.figure(figsize=(6,4))
    plt.bar(classifiers, times, color=['blue', 'green'])
    plt.ylabel('Tempo de Execução (s)')
    plt.title('Comparação de Tempo de Execução')
    plt.show()

def compare_classifiers():
    k_values = [1, 3, 5, 7]

    # Hardcore
    print("Executando o classificador Hardcore...")
    start_hc = time.time()
    hc_results = evaluate_multiple_k_values(k_values)
    hc_time = time.time() - start_hc

    # Sklearn
    print("\nExecutando o classificador Sklearn...")
    start_sklearn = time.time()
    sklearn_results = evaluate_multiple_k_values_sklearn(k_values)
    sklearn_time = time.time() - start_sklearn

    # Comparação textual
    print("\n--- Comparação de Tempos ---")
    print(f"Hardcore: {hc_time:.4f} s")
    print(f"Sklearn : {sklearn_time:.4f} s")

    print("\n--- Comparação de Métricas ---")
    for i, k in enumerate(k_values):
        print(f"\nk={k}:")
        print(f"  Hardcore -> Acurácia: {hc_results[i]['accuracy']:.4f}, Precisão: {hc_results[i]['precision']:.4f}, Revocação: {hc_results[i]['recall']:.4f}")
        print(f"  Sklearn  -> Acurácia: {sklearn_results[i]['accuracy']:.4f}, Precisão: {sklearn_results[i]['precision']:.4f}, Revocação: {sklearn_results[i]['recall']:.4f}")

    # Gráficos
    plot_metrics(k_values, hc_results, sklearn_results)
    plot_times(hc_time, sklearn_time)

if __name__ == "__main__":
    compare_classifiers()
