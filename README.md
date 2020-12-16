# Artificial-BRCA


## Project Plan: Synthesizing Genes for Privacy Preservation in Genomics Research

## Motivation
Advances in sequencing technologies have made it possible for individuals to get their DNA sequenced quickly and for a reasonable price.  One promise of bioinformatics research is to use  these sequences to predict medical conditions and health outcomes.  In order to realize this promise, researchers need real human genotypes and their associated phenotypes.  However, regulatory policies such as HIPAA in the United States and GDPR in the European Union protect individual privacy by preventing the sharing of any data that may identify any individual.  These two interests conflict, and that is the problem this research aims to resolve.

The goal of this research is to synthesize genomic data in such a way that the data is sufficiently noisy as to preclude re-identifying any individual yet not so noisy as to preclude meaningful scientific, biological research.

## Documents
Check out the presentation slides: https://docs.google.com/presentation/d/15bzOxeki0IRtcEN_kEqSboAFxWvh586WCLr9lKEM8qE/edit?usp=sharing

## Running the code


```
git clone https://github.com/ming94539/Artificial-BRCA
cd Artificial-BRCA
```
Recommend creating a virtual environment to keep your dependencies isolated. 
```
pip3 install -r artificial-genome-venv-req.txt
```
