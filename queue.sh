# first iteration
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
# python3 train.py -n large  -us -lr 0.0001 -e 50 -w 4 -c weighted 
# python3 train.py -n large  -us -lr 0.0001 -e 50 -w 4 -c normal 
# python3 train.py -n huge  -us -lr 0.0001 -e 50 -w 4 -c weighted 
# python3 train.py -n huge  -us -lr 0.0001 -e 50 -w 4 -c normal 
# python3 train.py -n huge  -us -lr 0.00003 -e 50 -w 4 -c dense -b 32 

# second iteration - searching around the small model
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c dense --d_model 16 --n_layers 8 --n_heads 8 --patch_size 4 -n custom -e 1
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c dense --d_model 32 --n_layers 8 --n_heads 8 --patch_size 4 -n custom -e 1
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c dense --d_model 64 --n_layers 8 --n_heads 8 --patch_size 4 -n custom -e 1
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c dense --d_model 96 --n_layers 8 --n_heads 8 --patch_size 4 -n custom -e 1
# python3 train.py -n small  -us -lr 0.0001 -e 50 -w 4 -c dense --d_model 128 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -n small  -us -lr 0.0001 -e 50 -w 4 -c normal

# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c dense --d_model 128 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -n small  -us -lr 0.0001 -e 50 -w 10 -c normal -wd 1e-06
# python3 train.py -n small  -us -lr 0.0001 -e 50 -w 10 -c normal -wd 1e-06 -ls 0
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c weighted --d_model 16 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c weighted --d_model 32 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c weighted --d_model 64 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c weighted --d_model 96 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c weighted --d_model 128 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c normal --d_model 16 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c normal --d_model 32 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c normal --d_model 64 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c normal --d_model 96 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c normal --d_model 128 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c feature_wise_weighted --d_model 128 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c feature_wise_weighted --d_model 96 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c bottlenecked_dense --d_model 128 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c bottlenecked_dense --d_model 96 --n_layers 8 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c bottlenecked_dense --d_model 128 --n_layers 12 --n_heads 12 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c bottlenecked_dense --d_model 192 --n_layers 12 --n_heads 12 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c bottlenecked_dense --d_model 156 --n_layers 12 --n_heads 12 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c bottlenecked_dense --d_model 64 --n_layers 12 --n_heads 8 --patch_size 4 -n custom
# python3 train.py -us -lr 0.0001 -e 50 -w 4 -c bottlenecked_dense --d_model 64 --n_layers 24 --n_heads 8 --patch_size 4 -n custom -wd 1e-6
python3 train.py -us -lr 0.0001 -e 50 -w 4 -c bottlenecked_dense --d_model 128 --n_layers 12 --n_heads 8 --patch_size 4 -n custom
