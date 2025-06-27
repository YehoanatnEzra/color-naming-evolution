# Final Project: Color-Naming Evolution

This repository contains the final project for Huji course Dynamics, Computation & Networks. Our aim was to critically examine the article **Efficient compression in color naming and its evolution**  by Zaslavsky, Kemp, Regier & Tishby.

![image](https://github.com/user-attachments/assets/18429d83-cf30-4796-a9dc-26578baaa5b5)

## Research Question
How do specific structural and typological characteristics of languages (e.g., number of basic color terms, focal color distributions, semantic domain granularity) influence their communicative fitness under the Information Bottleneck framework?
 
Sub-question:
- Random color space vs k-means color space vs Human made color space.
- How does the lexicon size affects the fitness of the language?
- How languages adapt to different environments?

---
## Our Project and Experiment

# Crowdsourced Human Benchmark
To ground our simulations in real behavior, we ran a color-naming survey with **50 participants**. Each person labeled **330 evenly-spaced CIELAB swatches** using one of the 11 basic English color terms (e.g., red, blue, green, etc.). In total we gathered **16,500 label assignments**, which we then:

1. **Aggregated** — for each swatch, counted how often each name was chosen.  
2. **Modeled** — computed the empirical probability of each swatch being labeled with each English color term:  

The resulting distribution defines our **Human-Derived Language**, used alongside the Random and Soft KMeans encoders for:

- **Fitness comparison** (β·Accuracy − Complexity)  
- **Confusion-matrix heatmaps**  
- **Lexicon-size and environment analyses**


# Different Environments
Unlike the original paper’s uniform sampling, we model five **artificial perceptual environments** with distinct color distributions to test how context shapes category systems:
1. **Uniform**  
   - Flat sampling over the full CIELAB gamut.  
2. **Beach**  
   - Emphasizes sandy yellows and ocean blues (e.g., C∗a clustered around warm, light hues).  
3. **Forest**  
   - Dense greens and browns dominate the distribution (mimicking foliage).  
4. **Urban**  
   - Centers on grays, neutrals, and occasional bright accents (street signs, buildings).  
5. **Sunset**  
   - Skews toward warm reds, oranges, and purples (twilight sky palette).

# Experiments Conducted

1. **Language Selection Showcase**  
   - Presented three representative language systems side by side:  
     - **Human-Derived Language:** built from our crowdsourced survey data.  
     - **Soft KMeans Language:** IB-inspired clustering encoder.  
     - **Random Baseline Language:** uniform random assignment.  
   - Confusion-matrix heatmaps and fitness curves for each are in `plots/`.

2. **Lexicon-Size Sweep**  
   - Measured fitness across lexica of size 2–20 for random, KMeans, and human-derived encoders.  

3. **Sharpness Variation**  
   - Fixed lexicon size, varied temperature to map the accuracy–complexity frontier.  

4. **Multi-Environment Analysis**  
   - Identified optimal category systems across uniform, beach, forest, urban, and sunset spaces.  

5. **Human Benchmark Comparison**  
   - Direct comparison of model confusion matrices to the crowdsourced human lexicon.  

> All figures, heatmaps, and detailed observations are available in `plots/` and the `Presentation.pptx`.

# Results & Takeaways

* **Structured encoders** (soft KMeans) consistently outperform random baselines across lexicon sizes.
* There exists a **sweet-spot lexicon size** balancing complexity and accuracy.
* **Environmental structure** (e.g., “forest” vs. “urban”) shifts optimal category boundaries.
* The simulated systems **approach** but do not fully match human crowd data—highlighting areas for future refinement.

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


