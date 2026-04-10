# Tool Analysis

## DeepMicroClass (DMC)
- Input: contig FASTA.
- Output used here: table with logits for `Eukaryote`, `EukaryoteVirus`, `Plasmid`, `Prokaryote`, `ProkaryoteVirus`, and `best_choice`.
- Role in the hybrid model: provides per-contig probability vectors (temperature-softmax) injected into graph nodes.

## 4CAC
- `classify_xgb.py`: produce score 4-class (`viral_score`, `plas_score`, `prokar_score`, `eukar_score`).
- `classify_4CAC.py`: uses the assembly graph (`assembly_graph_with_scaffolds.gfa`) + `scaffolds.paths` to:
  - assign initial classes to nodes,
  - run `correction` on classified nodes with concordant neighbors,
  - run `propagation` for uncertain nodes.
- Role in the hybrid model: topological propagation and local correction mechanism.

## Implemented Hybrid Strategy
1. Convert DMC logits into 4-class probabilities:
   - virus = `EukaryoteVirus + ProkaryoteVirus`
   - plasmid = `Plasmid`
   - bacteria = `Prokaryote`
   - eukaryote = `Eukaryote`
2. Select anchors ("certain" contigs) using a threshold on `max(prob)`.
  - Below-threshold contigs are marked uncertain but **keep their original DMC distribution**
    (not flattened to `0.25` per class).
  - Therefore, the anchor threshold mainly affects diagnostics (`anchors_n`, `anchors_rate`), not score rewriting before 4CAC.
3. Inject DMC probabilities into graph nodes using contig-length-weighted averaging.
4. Apply 4CAC-style correction + propagation on hard node labels.
5. Run probabilistic smoothing on nodes without hard labels.
6. Produce final contig decision from node-path aggregation.
7. Plasmid rescue: if a contig is graph-isolated and `p_plasmid > 0.6`, assign `Plasmid`.
