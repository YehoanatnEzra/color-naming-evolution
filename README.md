# Final Project: Color-Naming Evolution
This repository contains the final project for Huji course **67562: Dynamics, Computation & Networks**. Our aim was to critically examine the article  
> **“Efficient compression in color naming and its evolution”** by Zaslavsky, Kemp, Regier & Tishby.  
Specifically, we ask:  
- How do discrete color-naming systems emerge under the Information Bottleneck framework?  
- In what ways do replicator dynamics shape the balance between communicative efficiency (informativeness) and complexity (lexicon size)?  
- How closely can simulated languages reproduce real human color labels when benchmarked against crowdsourced English data?

## Research Question

How do human-like color-naming systems evolve under the Information Bottleneck framework combined with replicator dynamics, and how do they balance efficiency (informativeness) and complexity (lexicon size) across different perceptual environments?

---

## Project Overview

I implement a cutting-edge Python simulation that models the evolution of color-naming languages. My approach combines:

- **Information Bottleneck (IB):**  
  - Derive soft encoder–decoder mappings that compress continuous CIELAB color stimuli into discrete categories while preserving relevant perceptual information.  
- **Replicator Dynamics:**  
  - Evolve a population of languages over generations, where each language’s fitness is  
    \[
      \text{fitness} = \beta \times \text{Accuracy} \;-\; \text{Complexity}
    \]  
- **Human Benchmark:**  
  - Integrate crowdsourced English color-labeling data (participants labeled hundreds of CIELAB swatches) as a real-world lexicon to benchmark simulated results.

---

## Key Components

- **Information-Theory Core** (`utils.py`)  
  - Computes \(I(M;W)\) (complexity), \(I(M;U)\) (environmental information), accuracy, and overall fitness.  
- **Cognitive Environments** (`environment.py`)  
  - Define meaning spaces in CIELAB: uniform, beach, forest, urban, sunset.  
  - Configurable weighting functions to reflect natural color distributions.  
- **Encoder Strategies** (`encoder_strategies.py`)  
  - **Random Encoder:** Baseline uniform assignment.  
  - **Soft KMeans Encoder:** IB-inspired, clustering with adjustable “sharpness” (temperature).  
- **Language Model** (`language.py`)  
  - Represents a language by lexicon size, sharpness, and the conditional matrix \(Q(w|m)\).  
- **Simulation Loop** (`simulation.py`)  
  - Initializes populations (via `simulation_initializations.py`).  
  - Applies replicator updates over `NUM_GENERATIONS`.  
  - Logs intermediate and final population statistics.  
- **Experiments & Visualization** (`experiments.py`)  
  - Lexicon-size sweeps: KMeans vs. random.  
  - Sharpness variation with fixed lexicon.  
  - Multi-environment comparison.  
  - Human benchmark integration.  
  - Optional replicator–mutator extension.  
- **Configuration** (`config.py`)  
  - Central parameters: population size, β, lexicon ranges, temperature values, number of generations.  
- **Data & Plots**  
  - **`data/`** (git-ignored): raw CIELAB and crowdsourced CSV.  
  - **`plots/`**: all generated graphs (fitness curves, scatterplots, human vs. model comparisons).  
  - **`Presentation.pptx`**: full slide deck with results.

---

## Experiments Conducted

1. **Lexicon-Size Sweep**  
   - Measured fitness across lexica of size 2–20 for both soft KMeans and random encoders.  
2. **Sharpness Variation**  
   - Fixed lexicon size, varied temperature to explore the accuracy–complexity frontier.  
3. **Multi-Environment Analysis**  
   - Identified optimal category systems in uniform, beach, forest, urban, and sunset spaces.  
4. **Human Benchmark Comparison**  
   - Compared simulated lexica to crowdsourced English labels to gauge human-model alignment.  

> See the full set of graphs and qualitative observations in the Presentation (`Presentation.pptx`) and the `plots/` folder.

---

## Installation & Usage

```bash
git clone https://github.com/your-username/67562-dynamics-networks-computation.git
cd 67562-dynamics-networks-computation
pip install -r requirements.txt
````

* **Run the full simulation:**

  ```bash
  python simulation.py
  ```
* **Execute a specific experiment:**

  ```bash
  python experiments.py --experiment kmeans_vs_random
  ```

---

## Project Structure

```
.
├── environment.py
├── language.py
├── encoder_strategies.py
├── utils.py
├── simulation_initializations.py
├── simulation.py
├── experiments.py
├── config.py
├── data/                    # ignored: raw CIELAB and human CSV
├── plots/                   # generated figures
├── Presentation.pptx        # slide deck
├── requirements.txt
├── README.md                # this document
└── LICENSE
```

---

## Results & Takeaways

* **Structured encoders** (soft KMeans) consistently outperform random baselines across lexicon sizes.
* There exists a **sweet-spot lexicon size** balancing complexity and accuracy.
* **Environmental structure** (e.g., “forest” vs. “urban”) shifts optimal category boundaries.
* The simulated systems **approach** but do not fully match human crowd data—highlighting areas for future refinement.

---

## Extensibility

You can easily:

* Add new meaning spaces or import alternative perceptual datasets.
* Plug in different encoder/decoder strategies (e.g., Gaussian Mixtures).
* Explore replicator–mutator dynamics with custom mutation rates.
* Use the codebase to model other semantic domains beyond color.

---

## Citation

If you use this code, please cite:

> Zaslavsky, N., Kemp, C., Regier, T., & Tishby, N. (2018). *Efficient compression in color naming and its evolution*. PNAS.

---

## License & Contribution

This project is MIT-licensed. See `LICENSE`.
Contributions, issues, and pull requests are welcome—please read `CONTRIBUTING.md` for guidelines.

```
```
