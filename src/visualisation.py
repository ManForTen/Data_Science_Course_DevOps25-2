import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_plot(fig, path):
    """Сохраняем график и закрываем его"""
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_missing(data, save_path=None):
    """Визуализация пропусков"""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
    plt.title("Пропущенные значения", fontsize=16)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    

def plot_unique_counts(data, save_path=None):
    """Количество уникальных значений по признакам"""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axvline(x=data.shape[0], color='r', label='Среднее', linestyle='--', linewidth=1)
    ax.barh(data.columns, data.nunique(), align='center', color='lightgreen')
    ax.set_xlabel('Количество', fontsize=15)
    ax.set_ylabel('Признаки', fontsize=15)
    ax.set_title("Количество уникальных значений", fontsize=20)
    ax.legend()

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(data, save_path=None):
    """Матрица корреляции числовых признаков"""
    correlation_matrix = data.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Матрица корреляции", fontsize=16)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_numeric_distributions(data, save_dir=None):
    """Гистограммы и boxplots для числовых признаков"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.hist(data[col], bins=25, color='#087E8B')
        ax.set_title(f'Распределение {col}', fontsize=16)
        ax.set_xlabel(col)
        ax.set_ylabel('Количество')
        
        fig.savefig(os.path.join(save_dir, f"{col}_hist.png"), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.boxplot(x=data[col], ax=ax)
        ax.set_title(f'Боксплот {col}', fontsize=16)
        
        fig.savefig(os.path.join(save_dir, f"{col}_box.png"), bbox_inches="tight")
        plt.close(fig)


def plot_categorical_distributions(data, save_dir=None):
    """Барплоты для категориальных признаков"""
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=data[col].value_counts().index, y=data[col].value_counts(), ax=ax)
        ax.set_title(f'Распределение {col}', fontsize=16)
        ax.tick_params(axis='x', rotation=90)
        
        fig.savefig(os.path.join(save_dir, f"{col}_bar.png"), bbox_inches="tight")
        plt.close(fig)


def custom_hist(training_set, title, xlabel, ylabel='Количество', bins=25, save_path=None):
    """Гистограмма произвольного массива"""
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.hist(training_set, bins=bins, color='#087E8B')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
