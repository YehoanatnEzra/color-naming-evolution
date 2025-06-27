import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import utils
import encoder_strategies
import config
import numpy as np
import pandas as pd
import sys
sys.path.append(os.getcwd())
from environment import Environment


def test_compare_kmeans_vs_random_by_lexicon_size(env, env_name):
    """
    Compare KMeans vs Random encoders over a range of lexicon sizes,
    and plot all three metrics together in one figure,
    tagging the outputs with the given environment name.

    Args:
        env: An instance of Environment (any variant).
        env_name (str): A short label for this environment (e.g. "uniform", "beach").
    """
    # extract environment parameters
    cognitive_source = env.cognitive_source
    meaning_space = env.meaning_space
    sigma = env.sigma

    # precompute constants
    beta = config.BETA
    I_MU = utils.compute_I_MU(cognitive_source, meaning_space, sigma)
    sharpness = config.VARYING_KMEANS_SHARPNESS

    # sweep settings
    lexicon_sizes = list(range(2, 110, 2))
    results = {'kmeans': [], 'random': []}

    # collect metrics for each strategy and lexicon size
    for k in lexicon_sizes:
        for strat_name, strat_id in [('kmeans', 1), ('random', 0)]:
            encoder = encoder_strategies.get_encoder_by_strategy(
                strategy=strat_id,
                num_meanings=env.num_meanings,
                lexicon_size=k,
                meaning_space=meaning_space if strat_id == 1 else None,
                sharpness=sharpness if strat_id == 1 else None
            )
            c = utils.calculate_complexity(encoder, cognitive_source)
            a, _ = utils.calculate_accuracy(encoder, cognitive_source,
                                            meaning_space, sigma, I_MU)
            f = utils.calculate_fitness(encoder, cognitive_source,
                                        meaning_space, sigma, beta)
            results[strat_name].append((k, f, a, c))

    # build combined metrics dictionary
    metrics_dict = {
        'KMeans Fitness': [r[1] for r in results['kmeans']],
        'KMeans Accuracy': [r[2] for r in results['kmeans']],
        'KMeans Complexity': [r[3] for r in results['kmeans']],
        'Random Fitness': [r[1] for r in results['random']],
        'Random Accuracy': [r[2] for r in results['random']],
        'Random Complexity': [r[3] for r in results['random']],
    }

    # prepare filename and title
    filename = f"{env_name}_compare_all_metrics_vs_lexicon_size.png"
    title = f"All Metrics vs. Lexicon Size (KMeans vs Random) – {env_name}"

    # ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # plot and save
    _plot_parameter_sweep(
        x=lexicon_sizes,
        metrics_dict=metrics_dict,
        xlabel='Lexicon Size',
        title=title,
        filename=os.path.join('output', filename),
    )


def plot_kmeans_fitness_all_envs(env_specs):
    """
    For each (label, env) in env_specs, run KMeans (strategy=1)
    over lexicon sizes 2..110 and plot *only* fitness curves.
    """
    os.makedirs('output', exist_ok=True)

    lexicon_sizes = list(range(2, 110, 2))
    beta = config.BETA
    fitness_curves = {}

    for label, env in env_specs:
        cog   = env.cognitive_source
        ms    = env.meaning_space
        sigma = env.sigma
        I_MU  = utils.compute_I_MU(cog, ms, sigma)

        # collect fitness for KMeans at each k
        fitnesses = []
        for k in lexicon_sizes:
            encoder = encoder_strategies.get_encoder_by_strategy(
                strategy=1,
                num_meanings=env.num_meanings,
                lexicon_size=k,
                meaning_space=ms,
                sharpness=config.VARYING_KMEANS_SHARPNESS
            )
            f = utils.calculate_fitness(encoder, cog, ms, sigma, beta)
            fitnesses.append(f)

        fitness_curves[label] = fitnesses

    # plot only those fitness curves:
    _plot_parameter_sweep(
        x=lexicon_sizes,
        metrics_dict=fitness_curves,
        xlabel='Lexicon Size',
        title='KMeans Fitness vs. Lexicon Size Across Environments',
        filename=os.path.join('output','kmeans_fitness_all_envs.png'),
        ylabel="Fitness"
    )


def test_compare_three_english_variants(env):
    # 1) Load the human‐judgment file
    df = pd.read_csv("data/English.csv")

    # 2) Identify the percentage columns and parse them via regex
    perc_cols = df.columns[4:]
    perc = []
    for col in perc_cols:
        m = re.search(r'(\d+)%', col)
        if not m:
            raise ValueError(f"Cannot parse percentage from column name: {col}")
        perc.append(float(m.group(1)) / 100)

    # 3) Build the full “light/dark X” distribution (dist1)
    unique_terms = sorted({t for c in perc_cols for t in df[c].unique()})
    term_idx = {t:i for i,t in enumerate(unique_terms)}
    M, K1 = len(df), len(unique_terms)
    dist1 = np.zeros((M, K1))
    for i, row in df.iterrows():
        for col, p in zip(perc_cols, perc):
            dist1[i, term_idx[row[col]]] += p

    # 4) Collapse to base colors (dist2)
    bases = [t.split(' ',1)[-1] if ' ' in t else t for t in unique_terms]
    unique_bases = sorted(set(bases))
    base_idx = {b:i for i,b in enumerate(unique_bases)}
    K2 = len(unique_bases)
    dist2 = np.zeros((M, K2))
    for t, i_t in term_idx.items():
        b = t.split(' ',1)[-1] if ' ' in t else t
        dist2[:, base_idx[b]] += dist1[:, i_t]

    # 5) Shade‐only collapse (dist3)
    dist3 = np.zeros((M,2))  # 0=dark, 1=light
    for t, i_t in term_idx.items():
        if t == 'white' or t.startswith('light '):
            dist3[:,1] += dist1[:,i_t]
        else:
            dist3[:,0] += dist1[:,i_t]

    # 6) Load sim environment & compute metrics
    cog, ms = env.cognitive_source, env.meaning_space
    I_MU = utils.compute_I_MU(cog, ms, config.SIGMA)

    def metrics_for(enc):
        c = utils.calculate_complexity(enc, cog)
        a,_ = utils.calculate_accuracy(enc, cog, ms, config.SIGMA, I_MU)
        f = utils.calculate_fitness(enc, cog, ms, config.SIGMA, config.BETA)
        return c, a, f

    variants = {
        "distinction between shades": dist1,
        "No distinction between shades": dist2,
        "Shade only":      dist3,
    }

    # --- Print out metrics ---
    print(f"{'Variant':25s} {'Complexity':>12s} {'Accuracy':>12s} {'Fitness':>12s}")
    print("-"*64)
    for name, enc in variants.items():
        c, a, f = metrics_for(enc)
        print(f"{name:25s} {c:12.4f} {a:12.4f} {f:12.4f}")

    # --- INSERT PLOTTING HERE ---

    # 1) Define your variant labels and computed metrics:
    names = ["distinction between shades", "No distinction between shades", "Shade only"]
    complexities = [2.6547, 1.8617, 0.6911]  # replace with your dynamic values
    accuracies = [13.7194, 10.0281, 4.1586]
    fitnesses = [12.2995, 9.0689, 3.8417]

    # 2) Set up bar positions
    x = np.arange(len(names))
    width = 0.25

    # 3) Create the grouped bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, complexities, width, label='Complexity')
    plt.bar(x, accuracies, width, label='Accuracy')
    plt.bar(x + width, fitnesses, width, label='Fitness')

    # 4) Format axes and title
    plt.xticks(x, names)
    plt.ylabel('Metric Value')
    plt.title('Comparison of Complexity, Accuracy & Fitness by Variant')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 5) Final layout and display
    plt.tight_layout()
    plt.show()


def human_vs_kmean_vs_random(env):
    # --- 1) Human‐judgment variants from English.csv ---
    df = pd.read_csv("data/English.csv")
    perc_cols = df.columns[4:]
    # parse e.g. "20%" or "20%.1" → 0.20
    perc = []
    for col in perc_cols:
        m = re.search(r'(\d+)%', col)
        if not m:
            raise ValueError(f"Cannot parse percentage from column name: {col}")
        perc.append(int(m.group(1)) / 100)

    # Full light/dark lexicon
    unique_terms = sorted({t for c in perc_cols for t in df[c].unique()})
    idx_term = {t: i for i, t in enumerate(unique_terms)}
    M = len(df)
    dist1 = np.zeros((M, len(unique_terms)))
    for i, row in df.iterrows():
        for col, p in zip(perc_cols, perc):
            dist1[i, idx_term[row[col]]] += p

    # Base colors only (drop "light"/"dark")
    bases = [t.split(' ', 1)[-1] if ' ' in t else t for t in unique_terms]
    unique_bases = sorted(set(bases))
    idx_base = {b: i for i, b in enumerate(unique_bases)}
    dist2 = np.zeros((M, len(unique_bases)))
    for t, i_t in idx_term.items():
        b = t.split(' ', 1)[-1] if ' ' in t else t
        dist2[:, idx_base[b]] += dist1[:, i_t]

    # Shade only (2 words: dark vs light)
    dist3 = np.zeros((M, 2))  # col0=dark, col1=light
    for t, i_t in idx_term.items():
        if t == 'white' or t.startswith('light '):
            dist3[:, 1] += dist1[:, i_t]
        else:
            dist3[:, 0] += dist1[:, i_t]

    # --- 2) KMeans & Random variants at k=2,10,18 ---
    cog, ms = env.cognitive_source, env.meaning_space
    sigma = env.sigma

    def metrics_for(encoder):
        C = utils.calculate_complexity(encoder, cog)
        A, _ = utils.calculate_accuracy(encoder, cog, ms, sigma, utils.compute_I_MU(cog, ms, sigma))
        F = utils.calculate_fitness(encoder, cog, ms, sigma, config.BETA)
        return C, A, F

    sizes = [2, 10, 18]
    kmeans = [
        (f"KMeans k={k}",
         encoder_strategies.get_encoder_by_strategy(1, env.num_meanings, k,
                                                    meaning_space=ms,
                                                    sharpness=config.VARYING_KMEANS_SHARPNESS))
        for k in sizes
    ]
    randoms = [
        (f"Random k={k}",
         encoder_strategies.get_encoder_by_strategy(0, env.num_meanings, k))
        for k in sizes
    ]

    # --- 3) Collect all variants & compute metrics ---
    all_vars = [
                   ("Human Full", dist1),
                   ("Human No distinction between light and dark", dist2),
                   ("Human Shade", dist3),
               ] + kmeans + randoms

    results = []
    for name, enc in all_vars:
        C, A, F = metrics_for(enc)
        results.append((name, C, A, F))
    df_res = pd.DataFrame(results, columns=["Variant", "Complexity", "Accuracy", "Fitness"])

    # --- 4) Plotting ---
    out = "output"
    os.makedirs(out, exist_ok=True)

    x = np.arange(len(df_res))
    w = 0.6

    # Complexity
    plt.figure(figsize=(10, 5))
    plt.bar(x, df_res["Complexity"], w)
    plt.xticks(x, df_res["Variant"], rotation=45, ha="right")
    plt.ylabel("Complexity")
    plt.title("Complexity Across Variants")
    plt.tight_layout()
    plt.savefig(f"{out}/all_variants_complexity.png", dpi=300)
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.bar(x, df_res["Accuracy"], w)
    plt.xticks(x, df_res["Variant"], rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Variants")
    plt.tight_layout()
    plt.savefig(f"{out}/all_variants_accuracy.png", dpi=300)
    plt.close()

    # Fitness
    plt.figure(figsize=(10, 5))
    plt.bar(x, df_res["Fitness"], w)
    plt.xticks(x, df_res["Variant"], rotation=45, ha="right")
    plt.ylabel("Fitness")
    plt.title("Fitness Across Variants")
    plt.tight_layout()
    plt.savefig(f"{out}/all_variants_fitness.png", dpi=300)
    plt.close()

    # Grouped all three metrics in one
    plt.figure(figsize=(12, 6))
    bar_w = 0.25
    plt.bar(x - bar_w, df_res["Complexity"], bar_w, label="Complexity")
    plt.bar(x, df_res["Accuracy"], bar_w, label="Accuracy")
    plt.bar(x + bar_w, df_res["Fitness"], bar_w, label="Fitness")
    plt.xticks(x, df_res["Variant"], rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("All Metrics Across All Variants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out}/all_variants_grouped.png", dpi=300)
    plt.close()

    print("✅ Saved plots under output/")

def _plot_parameter_sweep(x, metrics_dict, xlabel, title, filename, ylabel = 'value'):
    """
    Plot multiple metrics (e.g., fitness, accuracy, complexity) vs. a parameter (x).
    metrics_dict: dict of {metric_name: list_of_values}
    """
    plt.figure()
    for metric_name, values in metrics_dict.items():
        plt.plot(x, values, marker='o', label=metric_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f'Plot saved: {filename}')


def load_human_variants(csv_path):
    """Load and build three human-judgment distributions from English.csv."""
    df = pd.read_csv(csv_path)
    perc_cols = df.columns[4:]
    perc = [float(re.search(r"(\d+)%", c).group(1)) / 100 for c in perc_cols]

    unique_terms = sorted({t for c in perc_cols for t in df[c].unique()})
    term_idx = {t: i for i, t in enumerate(unique_terms)}
    M = len(df)

    # full distribution (dist1)
    dist1 = np.zeros((M, len(unique_terms)))
    for i, row in df.iterrows():
        for col, p in zip(perc_cols, perc):
            dist1[i, term_idx[row[col]]] += p

    # collapse to base colors (dist2)
    bases = [t.split(' ', 1)[-1] if ' ' in t else t for t in unique_terms]
    base_set = sorted(set(bases))
    dist2 = np.zeros((M, len(base_set)))
    for idx, term in enumerate(unique_terms):
        base = term.split(' ', 1)[-1] if ' ' in term else term
        dist2[:, base_set.index(base)] += dist1[:, idx]

    # shade-only collapse (dist3)
    dist3 = np.zeros((M, 2))  # col0=dark, col1=light
    for idx, term in enumerate(unique_terms):
        if term == 'white' or term.startswith('light '):
            dist3[:, 1] += dist1[:, idx]
        else:
            dist3[:, 0] += dist1[:, idx]

    return ['Human full', 'Human no shades', 'Human only shades'], [dist1, dist2, dist3]


def plot_human_fitness(env, human_dists, out_path='output/human_fitness_comparison.png'):
    """Bar plot of fitness for the three human-judgment variants."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dist1, dist2, dist3 = human_dists
    labels = ['Human full', 'Human no shades', 'Human only shades']
    dists = [dist1, dist2, dist3]

    cog = env.cognitive_source
    ms = env.meaning_space
    sigma = env.sigma
    beta = config.BETA

    fitnesses = []
    for dist in dists:
        f = utils.calculate_fitness(dist, cog, ms, sigma, beta)
        fitnesses.append(f)

    x = np.arange(len(labels))
    plt.figure(figsize=(6,4))
    plt.bar(x, fitnesses, width=0.6)
    plt.xticks(x, labels, rotation=20, ha='right')
    plt.ylabel('Fitness')
    plt.title('Fitness Comparison: Human Variants')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved human fitness comparison to {out_path}")


def plot_kmeans_random_fitness(env, ks=[2,10,18,50,100,150], out_path='output/kmeans_random_fitness.png'):
    """Line plot of fitness vs. lexicon size for KMeans and Random strategies."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cog = env.cognitive_source
    ms = env.meaning_space
    sigma = env.sigma
    beta = config.BETA

    fitness_kmeans = []
    fitness_random = []
    for k in ks:
        enc_k = encoder_strategies.get_encoder_by_strategy(
            strategy=1,
            num_meanings=env.num_meanings,
            lexicon_size=k,
            meaning_space=ms,
            sharpness=config.VARYING_KMEANS_SHARPNESS
        )
        fitness_kmeans.append(utils.calculate_fitness(enc_k, cog, ms, sigma, beta))

        enc_r = encoder_strategies.get_encoder_by_strategy(
            strategy=0,
            num_meanings=env.num_meanings,
            lexicon_size=k
        )
        fitness_random.append(utils.calculate_fitness(enc_r, cog, ms, sigma, beta))

    plt.figure(figsize=(7,5))
    plt.plot(ks, fitness_kmeans, marker='o', label='KMeans')
    plt.plot(ks, fitness_random, marker='s', label='Random')
    plt.xlabel('Lexicon Size (k)')
    plt.ylabel('Fitness')
    plt.title('Fitness vs. Lexicon Size (KMeans vs Random)')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved KMeans vs Random fitness plot to {out_path}")


def run_replicator(env, variants, generations=None):
    """
    Run discrete-time replicator dynamic over given variants.
    Returns labels list and history array X of shape (len(variants), generations+1).
    """
    if generations is None:
        generations = config.NUM_GENERATIONS

    beta = config.BETA
    cog = env.cognitive_source
    ms = env.meaning_space
    sigma = env.sigma
    labels, encs = zip(*variants)
    M = len(encs)

    x = np.full(M, 1.0 / M)
    X = np.zeros((M, generations + 1))
    X[:, 0] = x

    for t in range(generations):
        f = np.zeros(M)
        for i, enc in enumerate(encs):
            f[i] = utils.calculate_fitness(enc, cog, ms, sigma, beta)
        phi = np.dot(x, f)
        if phi > 0:
            x = x * f / phi
            x /= x.sum()
        X[:, t + 1] = x

    return list(labels), X


def plot_population_share(labels, X, indices, out_path, title):
    """Plot population shares for selected variant indices over time with thinner lines."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    T = X.shape[1]

    plt.figure(figsize=(8, 5))
    for idx in indices:
        plt.plot(
            range(T),
            X[idx],
            marker='o',
            markersize=4,
            linewidth=1,
            label=labels[idx]
        )
    plt.xlabel('Generation')
    plt.ylabel('Population Share')
    plt.title(title)
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_population_dynamics(labels, X, out_path="output/population_dynamics.png"):
    """Plot each variant’s share over time."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    T = X.shape[1]

    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(range(T), X[i], lw=1, label=label)
    plt.xlabel("Generation")
    plt.ylabel("Population Share")
    plt.title("Evolution of Language Variants over Generations")
    plt.legend(ncol=2, fontsize="small", loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved population dynamics to {out_path}")

    def run_replicator(env, variants, generations=None, mu=0.01):
        """
        Discrete‐time replicator‐mutator dynamic over given variants.
        Adds a small mutation rate mu so that no strategy ever goes completely extinct.

        Args:
            env: An Environment instance.
            variants: List of (label, encoder_or_dist) pairs.
            generations: Number of generations to simulate (defaults to config.NUM_GENERATIONS).
            mu: Mutation rate per generation (default 1%).

        Returns:
            labels: List of variant names.
            X:      np.ndarray of shape (len(variants), generations+1) recording population shares.
        """
        import numpy as np
        import utils, config

        if generations is None:
            generations = config.NUM_GENERATIONS

        beta = config.BETA
        cog = env.cognitive_source
        ms = env.meaning_space
        sigma = env.sigma

        labels, encs = zip(*variants)
        M = len(encs)

        # initialize uniform shares
        x = np.full(M, 1.0 / M)
        X = np.zeros((M, generations + 1))
        X[:, 0] = x

        for t in range(generations):
            # compute fitness vector
            f = np.array([utils.calculate_fitness(enc, cog, ms, sigma, beta)
                          for enc in encs])

            phi = np.dot(x, f)
            if phi > 0:
                # replicator update
                x = x * f / phi
                # add small uniform mutation inflow
                x = (1 - mu) * x + mu * (1.0 / M)
                # renormalize
                x /= x.sum()

            X[:, t + 1] = x

        return list(labels), X

    def plot_population_share(labels, X, indices, out_path, title):
        """Plot population shares for selected variant indices over time."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        T = X.shape[1]

        plt.figure(figsize=(8, 5))
        for idx in indices:
            plt.plot(
                range(T), X[idx],
                marker='o', markersize=2,
                linewidth=1, label=labels[idx]
            )
        plt.xlabel('Generation')
        plt.ylabel('Population Share')
        plt.title(title)
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved plot: {out_path}")

    def compare_kmeans_random(env, ks_kmeans=[40, 45, 50, 55, 60, 65], ks_random=[],
                              out_path='output/kmeans_random_dynamics.png', generations=None):
        """
        Compare KMeans vs Random variants over time for specified k-values.
        Generates a population dynamics plot for these encoder strategies.
        """
        variants = []
        for k in ks_kmeans:
            variants.append((
                f'KMeans k={k}',
                encoder_strategies.get_encoder_by_strategy(
                    strategy=1,
                    num_meanings=env.num_meanings,
                    lexicon_size=k,
                    meaning_space=env.meaning_space,
                    sharpness=config.VARYING_KMEANS_SHARPNESS
                )
            ))
        for k in ks_random:
            variants.append((
                f'Random k={k}',
                encoder_strategies.get_encoder_by_strategy(
                    strategy=0,
                    num_meanings=env.num_meanings,
                    lexicon_size=k
                )
            ))
        labels, X = run_replicator(env, variants, generations)
        indices = list(range(len(variants)))
        plot_population_share(
            labels, X,
            indices=indices,
            out_path=out_path,
            title='Population Dynamics: KMeans vs Random'
        )

