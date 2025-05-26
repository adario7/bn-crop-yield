import gzip

SEPARATOR_CHAR = ';'
END_LINE = '\n'
LAST_COLUMN_KEEPED = 'Osservazione'

with gzip.open('data/crops.csv.gz', 'rt', encoding='utf-8') as reader:
    with gzip.open('build/outputs.csv.gz', 'wt', encoding='utf-8') as writer:
        
        line = reader.readline()
        cols = 0    
        for elem in line.split(SEPARATOR_CHAR):
            writer.write(elem)
            cols +=1
            if elem == LAST_COLUMN_KEEPED: break
            writer.write(SEPARATOR_CHAR)
        writer.write(END_LINE)

        line = reader.readline()
        while line != "":
            elems = line.split(SEPARATOR_CHAR)
            for i in range(cols -1):
                writer.write(elems[i] + SEPARATOR_CHAR)
            writer.write(elems[cols -1] + END_LINE)
            line = reader.readline()
