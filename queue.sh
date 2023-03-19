# python3 train.py -n base -us -lr 0.0001 -e 200 -w 10
# python3 train.py -n tiny  -us -lr 0.0001 -e 50 -w 4 -c dense 
# python3 train.py -n tiny  -us -lr 0.0001 -e 50 -w 4 -c weighted 
# python3 train.py -n tiny  -us -lr 0.0001 -e 50 -w 4 -c normal
# python3 train.py -n small  -us -lr 0.0001 -e 50 -w 4 -c dense 
# python3 train.py -n small  -us -lr 0.0001 -e 50 -w 4 -c weighted 
# python3 train.py -n small  -us -lr 0.0001 -e 50 -w 4 -c normal 
# python3 train.py -n base  -us -lr 0.0001 -e 50 -w 4 -c dense 
# python3 train.py -n base  -us -lr 0.0001 -e 50 -w 4 -c weighted 
# python3 train.py -n base  -us -lr 0.0001 -e 50 -w 4 -c normal 
# python3 train.py -n large  -us -lr 0.0001 -e 50 -w 4 -c dense 
python3 train.py -n large  -us -lr 0.0001 -e 50 -w 4 -c weighted 
python3 train.py -n large  -us -lr 0.0001 -e 50 -w 4 -c normal 
python3 train.py -n huge  -us -lr 0.0001 -e 50 -w 4 -c weighted 
python3 train.py -n huge  -us -lr 0.0001 -e 50 -w 4 -c normal 
python3 train.py -n huge  -us -lr 0.00003 -e 50 -w 4 -c dense -b 32 
