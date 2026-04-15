# Evolutionary Optimization & Algorithmic Analysis 🧬🤖


## 📌 Project Overview
This repository contains a deep-dive into **Evolutionary Algorithms (EAs)**, specifically focusing on the design and optimization of **Genetic Algorithms (GAs)** to solve complex mathematical landscapesThe project serves as a comprehensive benchmarking study comparing GAs against **Simulated Annealing (SA)** and **Random Hill Climbing (RHC)**

The goal was to move beyond basic implementations by utilizing **Hyperparameter Tuning (Grid Search)** to find the most efficient balance between solution quality (fitness) and computational cost (processing time)

---

## 🚀 Key Features
**Genetic Algorithm Framework:** Built from the ground up with custom classes for `Individuals` and `Populations`.
* **Advanced Evolutionary Operators:**
    * **Selection:** Implemented Elitism-based selection to ensure the best candidates survive.
    * **Crossover:** Utilized Two-Point Crossover to explore new regions of the solution space.
    * **Mutation:** Dynamic mutation logic with tunable rates and step sizes to prevent stagnation in local optima.
* **Hyperparameter Optimization:** Integrated a **Grid Search** method to automate the testing of different generation counts, mutation rates, and step sizes.
* **Data Visualization:** Automated 3D scatter plotting to visualize the fitness landscape and convergence trends.

---

## 📊 Benchmarking & Performance
The project rigorously tested algorithms against complex benchmarks like the **Schwefel function**, known for its many local minima[cite: 429, 436].

### Algorithm Comparison Summary:
| Algorithm | Final Fitness (Lower is Better) | Strength |
| :--- | :--- | :--- |
| **Genetic Algorithm (GA)** | **4.57** |Superior exploration; avoided local optima. |
| **Simulated Annealing (SA)** | 5398.73 | Effective probabilistic escape from local minima. |
| **Random Hill Climbing (RHC)** | 37256.54 | Highly efficient but prone to "greedy" stagnation. |

---

## 🛠 Tech Stack
* **Language:** Python
* **Math & Data:** NumPy (for complex fitness calculations) 
* **Visualization:** Matplotlib (for 3D scatter plots and convergence graphs) 
* **Methodology:** Grid Search, Comparative Analysis 

---

## 📂 Repository Structure
```text
├── main.py             # Core execution script with Grid Search logic
├── ga_logic.py         # Genetic Algorithm implementation (Individual/Population classes)
├── sa_logic.py         # Simulated Annealing implementation
├── rhc_logic.py        # Random Hill Climbing implementation
├── functions.py        # Benchmark functions (Schwefel, etc.)
├── requirements.txt    # Project dependencies
└── results/            # Saved 3D scatter plots and performance graphs
```

---

## 📖 Key Learnings
1.  **The Generation Trade-off:** Increasing generations generally improves fitness (e.g., reaching **0.7139** at 800 generations), but yields diminishing returns relative to the increased processing time (75.3s vs 20.2s for 500 generations).
2.  **Diversity is Key:** High mutation rates (0.8+) were critical in maintaining population diversity and reaching near-global optima in complex search spaces.
3.  **Optimization Efficiency:** Carefully tuned parameters can produce high-quality solutions significantly faster than "brute-force" long-running generations.

## Contact

Selin Vural - https://www.linkedin.com/in/selinvrl/
Project Link: https://github.com/slnsuvrl/Evolutionary-Algorithm
