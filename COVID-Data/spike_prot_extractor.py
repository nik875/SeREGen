"""
Extracting S surface glycoprotein (spike protein) gene from COVID-19 sequences.
"""
from tqdm import tqdm as tqdm
from Bio import SeqIO
REF_SEQ_LOWER_BOUND = 21563
REF_SEQ_UPPER_BOUND = 25384


if __name__ == '__main__':
    records = list(tqdm(SeqIO.parse('sequences.fasta', 'fasta')))
    for i in tqdm(records):
        i.seq = i.seq[REF_SEQ_LOWER_BOUND:REF_SEQ_UPPER_BOUND]
    SeqIO.write(records, 'spike_prots.fasta', 'fasta')

