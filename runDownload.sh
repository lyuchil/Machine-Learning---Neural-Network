#!/bin/bash

module load python/3.11.6 py-requests zstd
echo "Starting Download"
pOut=$(python downloadPGN.py $1 $2)

successOut="Success!" 

if [ "$pOut" = "$successOut" ]; then
    filename="lichess_db_standard_rated_$1-$2.pgn"
    echo $filename
    mv $filename.zst ./compressed/
    echo "decompressing file!"
    unzstd -d ./compressed/$filename.zst -o ./rawGames/lcdb_$1-$2.pgn
else
    echo "Error in downloadPGN.py: $pOut"
fi

