import MySQLdb
import re
import pandas as pd


def get_chromosomes_from_UCSC():
    """
     Helper function, downloads the chromosome information for all the
     predefined species and saves them to the bins folder.

     The resulting files were manually checked against NCBI and modified
     where needed: reordering Roman numbers, etc..

     The actual writing line is commented out to prevent overwriting the
     correct, manually checked files. So uncomment it if needed.
    """
    # species to check in the UCSC genome browser for chromosomal info
    species = {
        'Human': 'hg38',
        'Mouse': 'mm10',
        'C. elegans': 'ce10',
        'D. melanogaster': 'dm3',
        'S. cerevisiae': 'sacCer3',
        'Chimp': 'panTro4',
        'Gorilla': 'gorGor3',
        'Orangutan': 'ponAbe2',
        'Rhesus': 'rheMac3',
        'Rabbit': 'oryCun2',
        'Pig': 'susScr3',
        'Sheep': 'oviAri3',
        'Cow': 'bosTau7',
        'Horse': 'equCab2',
        'Dog': 'canFam3',
        'Cat': 'felCat5',
        'Chicken': 'galGal4',
        'Turkey': 'melGal1'
    }

    species_no_to_check = ['C. elegans', 'D. melanogaster',
                           'S. cerevisiae', 'Ebola virus']
    for s in species.keys():
        print s
        db = MySQLdb.connect(host="genome-mysql.cse.ucsc.edu",
                             user="genomep",
                             passwd="password",
                             db=species[s])

        query = "select chrom,size from chromInfo limit 50"
        chromo = pd.read_sql(query, db)
        db.close()
        chrs = {}
        for ci, c in enumerate(chromo.chrom.values):
            if s not in species_no_to_check:
                if re.search(r'[^chr0-9ABCDEFXYZW]', c) == None:
                    try:
                        chrs[int(re.search(r'chr([0-9ABCDEFXYZW]{1,2})', c).group(1))] = int(chromo.iloc[ci, 1])
                    except:
                        chrs[re.search(r'chr([0-9ABCDEFXYZW]{1,2})', c).group(1)] = int(chromo.iloc[ci, 1])
            else:
                chrs[c.replace('chr', '')] = int(chromo.iloc[ci, 1])
                # write to files
        '''
        o = open('../Bins/'+s+'_chromosomes2.txt','w')
        for c in sorted(chrs):
            o.write(str(c)+'\t'+str(chrs[c])+'\n')
        o.close()
        print chrs
        '''