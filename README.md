# Chess-Fragility-GNN-Autoencoder-Analysis
**Contributers:** Sujayanand Kingsly & Siddarth Bhupathiraju  

## Aim
Fragility is the measure of the tension within a position and also an indicator of the tipping point in a game of chess.  
This work aims to see if a Graph Neural Network and Autoencoder combination can intuitively learn this concept or if it learns something else entirely.  
Our work is based off of [Marc Barthelemy's](https://arxiv.org/abs/2410.02333) research.  

## How to run
Simply pull a local copy of this repository, install the requirements in the "requirements.txt" file and run the "gnn_solution.py" file under the src folder.

## Architecture
<img width="917" height="323" alt="GNN-Autoencoder" src="https://github.com/user-attachments/assets/c6730aa4-8959-40bc-b9a5-4c2fd61ea3c1" />

The role of the autoencoder is to reconstruct the chess piece interaction graph and the role of the GNN is to represent the pieces on the chess board.

## Results
The GNN-Autoencoder combination learns the concept of fragility when Harmonic and Closeness centralities is used rather than betweenness centrality which is used by the author whose work this project is based on.  
Larger scale experiments will be required to solidify this claim.
