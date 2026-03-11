# Hybrid + Rescue: classificazione metagenomica multiclasse

Questo repository contiene il codice sorgente per il modello computazionale **Hybrid + Rescue**, sviluppato per migliorare la classificazione metagenomica multiclasse.

Il progetto supera i limiti intrinseci degli strumenti attuali combinando due paradigmi ortogonali:
1. L'estrazione semantica tramite **DeepMicroClass (DMC)**.
2. La propagazione topologica su grafo di assemblaggio tramite **4CAC**.

A questa architettura è integrata un'euristica originale, denominata **Plasmid Rescue**, progettata per recuperare sistematicamente i *contig* plasmidici che, a causa della loro natura di nodi isolati nel grafo, verrebbero persi dai vincoli rigidi della propagazione spaziale.

## Struttura del repository

- `scripts/`: codice sorgente principale.
  - `hybrid_dmc_4cac.py`: architettura (integrazione DMC-4CAC, calcolo metriche, implementazione *Plasmid Rescue*).
  - `evaluate_sharon_strategies.py`: script per il benchmarking sul dataset ambientale.
- `config/`: file JSON di configurazione per i percorsi di input.
- `docs/`: analisi dei risultati sul dataset Sharon.
- `results/`

## Requisiti e installazione

Il progetto è sviluppato in Python. Si raccomanda l'utilizzo di un ambiente virtuale (es. Conda).

**Dipendenze principali:**
- `numpy`
- `pandas`
- `scikit-learn`
- Ambiente configurato per l'esecuzione di **4CAC** (indicabile tramite il flag `--fourcac-env`).

## Utilizzo ed esecuzione

Lo script principale `hybrid_dmc_4cac.py` gestisce l'intera architettura. Di seguito i comandi principali:

### 1. Confronto baseline (4CAC vs DMC)
Valuta le prestazioni isolate dei due modelli di base prima dell'integrazione ibrida.
```
python scripts/hybrid_dmc_4cac.py compare-baseline \
  --c4-file data/output/4cac/sharon/4CAC_classification.fasta \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/baseline
```

### 2. Ottimizzazione parametri (Grid Search)
Esegue una ricerca per valutare l'impatto delle soglie sulle metriche di classificazione.
```
python scripts/hybrid_dmc_4cac.py grid-search \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gfa-file data/output/metaspades/sharon/assembly_graph_with_scaffolds.gfa \
  --paths-file data/output/metaspades/sharon/scaffolds.paths \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/grid_search \
  --fourcac-script path/to/4CAC/run_4CAC.py \
  --asmdir data/output/metaspades/sharon/ \
  --anchor-thresholds 0.55:0.95:0.05 \
  --plasmid-rescue-threshold 0.6 \
  --temperature 1.0 
```
> **Nota:** nell'implementazione finale, la propagazione sul grafo è delegata nativamente a 4CAC, che adotta una soglia di sbarramento interna severa (0.95) per le classi minoritarie. Il parametro `--anchor-threshold` serve in questa fase primariamente per la reportistica e la diagnostica (conteggio e *rate* delle ancore originali), senza forzare l'alterazione del vettore stocastico in ingresso per i nodi incerti.

## Output principali

Al termine dell'esecuzione, la cartella di output conterrà i seguenti file:
- `predictions_hybrid.tsv`: l'etichetta tassonomica finale assegnata a ciascun *contig*.
- `dmc_probabilities.tsv`: i vettori Softmax normalizzati estratti da DeepMicroClass.
- `metrics_hybrid.tsv`: risultati prestazionali (Precision, Recall, F1-score globali e per singola classe).
- `grid_search_summary.tsv` *(solo in modalità grid-search)*: tabella comparativa delle configurazioni testate.

## Riproducibilità

Il repository include l'infrastruttura per replicare i test su metagenomi simulati *in silico*, al fine di valutare la robustezza del modello su tecnologie di sequenziamento di seconda e terza generazione.

La pipeline è gestita tramite **SLURM** e genera due rami di analisi:
- **Ramo Short-read:** simulazione metagenomica con `ART` e assemblaggio tramite `metaSPAdes --meta`.
- **Ramo Long-read:** simulazione tramite `NanoSim3` e assemblaggio con `Flye --meta`.

**Avvio della pipeline su cluster:**
```
bash slurm/contig_scenarios/submit_workflow.sh
```

I risultati, comprensivi di grafici comparativi per i vari scenari ecologici (dominanza batterica, bilanciato, arricchimento virale/plasmidico), verranno salvati in:
- `validazione/camisim_contig_scenarios/results_short/` (e relativi grafici in `plots_short/`)
- `validazione/camisim_contig_scenarios/results_long/` (e relativi grafici in `plots_long/`)

---
*Lavoro sviluppato come tesi per il Corso di Laurea Magistrale in Biotecnologie Industriali, Università degli Studi di Padova (A.A. 2025/2026).*
