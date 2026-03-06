# Analisi strumenti

## DeepMicroClass (DMC)
- Input: contig FASTA.
- Output usato qui: tabella con logit per `Eukaryote`, `EukaryoteVirus`, `Plasmid`, `Prokaryote`, `ProkaryoteVirus` e `best_choice`.
- Ruolo nel modello ibrido: fornisce vettori probabilistici per ogni contig (softmax con temperatura) da iniettare nei nodi del grafo.

## 4CAC
- `classify_xgb.py`: produce score 4-class (`viral_score`, `plas_score`, `prokar_score`, `eukar_score`).
- `classify_4CAC.py`: usa grafo di assemblaggio (`assembly_graph_with_scaffolds.gfa`) + `scaffolds.paths` per:
  - assegnare classi iniziali ai nodi,
  - fare `correction` su nodi classificati con vicini concordi,
  - fare `propagation` ai nodi incerti.
- Ruolo nel modello ibrido: meccanismo di propagazione topologica e correzione locale.

## Strategia ibrida implementata
1. Conversione dei logit DMC in probabilità 4-class:
   - virus = `EukaryoteVirus + ProkaryoteVirus`
   - plasmide = `Plasmid`
   - batterio = `Prokaryote`
   - eucariote = `Eukaryote`
2. Selezione anchor (contig "certi") con soglia su `max(prob)`.
  - I contig sotto soglia sono marcati come incerti, ma **mantengono la distribuzione DMC originale**
    (non vengono uniformati a 0.25 per classe).
  - Quindi la soglia anchor impatta principalmente la diagnostica (`anchors_n`, `anchors_rate`) e non una riscrittura dei punteggi in input a 4CAC.
3. Iniezione delle probabilità DMC nei nodi del grafo tramite media pesata sulla lunghezza dei contig.
4. Correzione + propagazione in stile 4CAC su etichette hard dei nodi.
5. Smoothing probabilistico sui nodi non hard-labeled.
6. Decisione finale per contig da nodi del path.
7. Plasmid rescue: se contig isolato nel grafo e `p_plasmid > 0.6`, assegna `Plasmid`.
