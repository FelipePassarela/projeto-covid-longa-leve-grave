import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(path, score, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))

    for file in os.listdir(path):
        if file.endswith(".csv"):
            model_name = file.split(".")[0]
            data = pd.read_csv(os.path.join(path, file))
            plt.plot(data["n_features"], data[score], label=model_name, marker='o')

    plt.title(title)
    plt.xticks(data["n_features"])
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title.replace(" ", "_").lower()}.png")


def plot_figure_with_all_plots():
    dirs = {
        'test/standart': '../results/test/standart',
        'train/standart': '../results/train/standart',
        'test/tuned': '../results/test/tuned',
        'train/tuned': '../results/train/tuned',
    }

    dirs_titles = {
        'test/standart': 'Teste (Padrão)',
        'train/standart': 'Treino (Padrão)',
        'test/tuned': 'Teste (Ajustado)',
        'train/tuned': 'Treino (Ajustado)',
    }

    # Reading all csv files and concatenating them
    data_list = []
    for key, value in dirs.items():
        for file in os.listdir(value):
            if file.endswith(".csv"):
                data = pd.read_csv(os.path.join(value, file))
                data["model"] = file.split(".")[0]
                data["type"] = key
                data_list.append(data)

    data = pd.concat(data_list, ignore_index=True)
    data["n_features"] = data["n_features"].astype(str)
    
    main_title = "Comparação Padrão vs Ajustado (Treino e Teste)"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    fig.suptitle(main_title, fontsize=18)

    # Plotting roc_auc for all dirs type
    for i, (key, value) in enumerate(dirs.items()):
        ax = axes[i // 2, i % 2]
        for model in data["model"].unique():
            model_data = data[(data["model"] == model) & (data["type"] == key)]
            ax.plot(model_data["n_features"], model_data["roc_auc"], label=model, marker='o')
        ax.set_title(dirs_titles[key])
        ax.set_ylim(0.5, 1.025)
        ax.grid(True, which='both', linestyle='--')
        ax.legend()
        
        if i == 0 or i == 1:
            ax.tick_params(axis='x', which='both', length=0)
        if i == 1 or i == 3:
            ax.tick_params(axis='y', which='both', length=0) 
    
    fig.text(0.5, 0.02, 'Número de SNPs', ha='center', va='center', fontsize=14)
    fig.text(0.03, 0.5, 'ROC AUC', ha='center', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.savefig(f"../plots/{main_title.replace(' ', '_').lower()}.png")
    plt.show()
    
if __name__ == "__main__":
    plot_figure_with_all_plots()                
    