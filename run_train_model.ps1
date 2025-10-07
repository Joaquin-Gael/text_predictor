ECHO "start training model"

poetry run py train.py train -csp "corpus-2025-07-10_19-20-14-tokens_100-rows_107701.csv" -hs 1000 -es 1000 -d 0.3 -lr 0.001 -b 64 -e 5