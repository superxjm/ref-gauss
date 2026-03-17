#  python train.py -s data/ref_nerf/coffee --eval --white_background   
#  python train.py -s data/ref_nerf/helmet --eval  --white_background  --lambda_normal_smooth 1.0
#  python train.py -s data/ref_nerf/ball --eval  --white_background --lambda_normal_smooth 1.0 
#  python train.py -s data/ref_nerf/teapot --eval  --white_background 
#  python train.py -s data/ref_nerf/toaster --eval  --white_background   
#  python train.py -s data/ref_nerf/car --eval  --white_background 

#  python train.py -s data/GlossySynthetic/angel_blender --eval --white_background   
#  python train.py -s data/GlossySynthetic/potion_blender --eval  --white_background   
#  python train.py -s data/GlossySynthetic/horse_blender --eval  --white_background   
#  python train.py -s data/GlossySynthetic/luyu_blender --eval  --white_background    
#  python train.py -s data/GlossySynthetic/teapot_blender --eval  --white_background 
#  python train.py -s data/GlossySynthetic/bell_blender --eval  --white_background   
#  python train.py -s data/GlossySynthetic/tbell_blender --eval  --white_background  --lambda_normal_smooth 1.0
#  python train.py -s data/GlossySynthetic/cat_blender --eval  --white_background 


#  python train.py -s ../gardenspheres \
#     --eval --iterations 20000 \
#     --indirect_from_iter 10000 \
#     --volume_render_until_iter 0 \
#     --initial 1 \
#     --init_until_iter 6000 \
#     --lambda_normal_smooth 0.45 \
#     -r 4 

#  python train.py -s data/ref_real/toycar --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4
#  python train.py -s data/ref_real/sedan --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 8 

# python train.py -s blender_data_2 \
#     --eval \
#     --iterations 30000 \
#     --indirect_from_iter 10000 \
#     --volume_render_until_iter 0 \
#     --initial 1 \
#     --init_until_iter 3000 \
#     --lambda_normal_smooth 0.45 \
#     -r 4

# python train.py -s ../blender_data_2 \
#     --eval \
#     --iterations 20000 \
#     --indirect_from_iter 10000 \
#     --volume_render_until_iter 0 \
#     --initial 1 \
#     --init_until_iter 5000 \
#     --lambda_normal_smooth 0.45 \
#     -r 2

# python train.py -s ../kitchen_fipt \
#     --eval \
#     --iterations 20000 \
#     --indirect_from_iter 10000 \
#     --volume_render_until_iter 0 \
#     --initial 1 \
#     --init_until_iter 6000 \
#     --lambda_normal_smooth 0.45 \
#     -r 2

python train.py -s ../kitchen_flash \
    --eval \
    --iterations 30000 \
    --indirect_from_iter 10000 \
    --volume_render_until_iter 0 \
    --initial 1 \
    --init_until_iter 6000 \
    --lambda_normal_smooth 0.45 \
    -r 2